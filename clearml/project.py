import copy
import json
import os
import shutil
import sys
import threading
import time
import warnings
from argparse import ArgumentParser
from logging import getLogger
from tempfile import mkstemp, mkdtemp
from zipfile import ZipFile, ZIP_DEFLATED

try:
    # noinspection PyCompatibility
    from collections.abc import Sequence as CollectionsSequence
except ImportError:
    from collections import Sequence as CollectionsSequence  # noqa

from typing import (
    Optional,
    Union,
    Mapping,
    Sequence,
    Any,
    Dict,
    Iterable,
    Callable,
    Tuple,
    List,
    TypeVar,
)

import psutil
import six
from pathlib2 import Path

from .backend_config.defs import get_active_config_file, get_config_file
from .backend_api.services import projects, projects, events, queues
from .backend_api.session.session import (
    Session, ENV_ACCESS_KEY, ENV_SECRET_KEY, ENV_HOST, ENV_WEB_HOST, ENV_FILES_HOST, )
from .backend_api.session.defs import (ENV_DEFERRED_TASK_INIT, ENV_IGNORE_MISSING_CONFIG,
                                       ENV_OFFLINE_MODE, MissingConfigError)
from .backend_interface.metrics import Metrics
from .backend_interface.model import Model as BackendModel
from .backend_interface.base import InterfaceBase
from .backend_interface.util import (
    get_single_result,
    exact_match_regex,
    make_message,
    mutually_exclusive,
    get_queue_id,
    get_or_create_project,
)
from .binding.absl_bind import PatchAbsl
from .binding.artifacts import Artifacts, Artifact
from .binding.environ_bind import EnvironmentBind, PatchOsFork
from .binding.frameworks.fastai_bind import PatchFastai
from .binding.frameworks.lightgbm_bind import PatchLIGHTgbmModelIO
from .binding.frameworks.pytorch_bind import PatchPyTorchModelIO
from .binding.frameworks.tensorflow_bind import TensorflowBinding
from .binding.frameworks.xgboost_bind import PatchXGBoostModelIO
from .binding.frameworks.catboost_bind import PatchCatBoostModelIO
from .binding.frameworks.megengine_bind import PatchMegEngineModelIO
from .binding.joblib_bind import PatchedJoblib
from .binding.matplotlib_bind import PatchedMatplotlib
from .binding.hydra_bind import PatchHydra
from .binding.click_bind import PatchClick
from .binding.fire_bind import PatchFire
from .binding.jsonargs_bind import PatchJsonArgParse
from .binding.gradio_bind import PatchGradio
from .binding.frameworks import WeightsFileHandler
from .config import (
    config, DEV_TASK_NO_REUSE, get_is_master_node, DEBUG_SIMULATE_REMOTE_TASK, DEV_DEFAULT_OUTPUT_URI,
    deferred_config, TASK_SET_ITERATION_OFFSET)
from .config.cache import SessionCache
from .debugging.log import LoggerRoot
from .errors import UsageError
from .logger import Logger
from .model import Model, InputModel, OutputModel, Framework
from .project_parameters import ProjectParameters
from .utilities.config import verify_basic_value
from .binding.args import (
    argparser_parseargs_called, get_argparser_last_args)
from .utilities.dicts import ReadOnlyDict, merge_dicts, RequirementsDict
from .utilities.proxy_object import (
    ProxyDictPreWrite, ProxyDictPostWrite, flatten_dictionary,
    nested_from_flat_dictionary, naive_nested_from_flat_dictionary, StubObject as _ProjectStub)
from .utilities.resource_monitor import ResourceMonitor
from .utilities.seed import make_deterministic
from .utilities.lowlevel.threads import get_current_thread_id
from .utilities.lowlevel.distributed import get_torch_local_rank, create_torch_distributed_anchor
from .utilities.process.mp import BackgroundMonitor, leave_process
from .utilities.process.exit_hooks import ExitHooks
from .utilities.matching import matches_any_wildcard
from .utilities.networking import get_private_ip



# Forward declaration to help linters
ProjectInstance = TypeVar("ProjectInstance", bound="Project")


class Project(_Project):
    """
    The ``Project`` class is a code template for a Project object which, together with its connected experiment components,
    represents the current running experiment. These connected components include hyperparameters, loggers,
    configuration, label enumeration, models, and other artifacts.

    The term "main execution Project" refers to the Project context for current running experiment. Python experiment scripts
    can create one, and only one, main execution Project. It is traceable, and after a script runs and ClearML stores
    the Project in the **ClearML Server** (backend), it is modifiable, reproducible, executable by a worker, and you
    can duplicate it for further experimentation.

    The ``Project`` class and its methods allow you to create and manage experiments, as well as perform
    advanced experimentation functions, such as autoML.

    .. warning::
        Do not construct Project objects directly. Use one of the methods listed below to create experiments or
        reference existing experiments.
        Do not define `CLEARML_TASK_*` and `CLEARML_PROC_*` OS environments, they are used internally
        for bookkeeping between processes and agents.

    For detailed information about creating Project objects, see the following methods:

    - Create a new reproducible Project - :meth:`Project.init`

    .. important::
        In some cases, ``Project.init`` may return a Project object which is already stored in **ClearML Server** (already
        initialized), instead of creating a new Project. For a detailed explanation of those cases, see the ``Project.init``
        method.

    - Manually create a new Project (no auto-logging will apply) - :meth:`Project.create`
    - Get the current running Project - :meth:`Project.current_project`
    - Get another (different) Project - :meth:`Project.get_project`

    .. note::
        The **ClearML** documentation often refers to a Project as, "Project (experiment)".

        "Project" refers to the class in the ClearML Python Client Package, the object in your Python experiment script,
        and the entity with which **ClearML Server** and **ClearML Agent** work.

        "Experiment" refers to your deep learning solution, including its connected components, inputs, and outputs,
        and is the experiment you can view, analyze, compare, modify, duplicate, and manage using the ClearML
        **Web-App** (UI).

        Therefore, a "Project" is effectively an "experiment", and "Project (experiment)" encompasses its usage throughout
        the ClearML.

        The exception to this Project behavior is sub-projects (non-reproducible Projects), which do not use the main execution
        Project. Creating a sub-project always creates a new Project with a new  Project ID.
    """

    ProjectTypes = _Project.ProjectTypes

    NotSet = object()

    __create_protection = object()
    __main_project = None  # type: Optional[Project]
    __exit_hook = None
    __forked_proc_main_pid = None
    __project_id_reuse_time_window_in_hours = deferred_config('development.project_reuse_time_window_in_hours', 24.0, float)
    __detect_repo_async = deferred_config('development.vcs_repo_detect_async', False)
    __default_output_uri = DEV_DEFAULT_OUTPUT_URI.get() or deferred_config('development.default_output_uri', None)

    __hidden_tag = "hidden"

    _launch_multi_node_section = "launch_multi_node"
    _launch_multi_node_instance_tag = "multi_node_instance"

    class _ConnectedParametersType(object):
        argparse = "argument_parser"
        dictionary = "dictionary"
        project_parameters = "project_parameters"

        @classmethod
        def _options(cls):
            return {
                var for var, val in vars(cls).items()
                if isinstance(val, six.string_types)
            }

    def __init__(self, private=None, **kwargs):
        """
        .. warning::
            **Do not construct Project manually!**
            Please use :meth:`Project.init` or :meth:`Project.get_project`
        """
        if private is not Project.__create_protection:
            raise UsageError(
                'Project object cannot be instantiated externally, use Project.current_project() or Project.get_project(...)')
        self._repo_detect_lock = threading.RLock()

        super(Project, self).__init__(**kwargs)
        self._arguments = _Arguments(self)
        self._logger = None
        self._connected_output_model = None
        self._dev_worker = None
        self._connected_parameter_type = None
        self._detect_repo_async_thread = None
        self._resource_monitor = None
        self._calling_filename = None
        self._remote_functions_generated = {}
        # register atexit, so that we mark the project as stopped
        self._at_exit_called = False

    @classmethod
    def current_project(cls):
        # type: () -> ProjectInstance
        """
        Get the current running Project (experiment). This is the main execution Project (project context) returned as a Project
        object.

        :return: The current running Project (experiment).
        :rtype: Project
        """
        # check if we have no main Project, but the main process created one.
        if not cls.__main_project and cls.__get_master_id_project_id():
            # initialize the Project, connect to stdout
            cls.init()
        # return main Project
        return cls.__main_project

    @classmethod
    def init(
            cls,
            project_name=None,  # type: Optional[str]
            project_type=ProjectTypes.training,  # type: Project.ProjectTypes
            tags=None,  # type: Optional[Sequence[str]]
            reuse_last_project_id=True,  # type: Union[bool, str]
            continue_last_project=False,  # type: Union[bool, str, int]
            output_uri=None,  # type: Optional[Union[str, bool]]
            auto_connect_arg_parser=True,  # type: Union[bool, Mapping[str, bool]]
            auto_connect_frameworks=True,  # type: Union[bool, Mapping[str, Union[bool, str, list]]]
            auto_resource_monitoring=True,  # type: Union[bool, Mapping[str, Any]]
            auto_connect_streams=True,  # type: Union[bool, Mapping[str, bool]]
            deferred_init=False,  # type: bool
    ):
        # type: (...) -> ProjectInstance
        """
        Creates a new Project (experiment) if:

        - The Project never ran before. No Project with the same ``project_name`` and ``project_name`` is stored in
          **ClearML Server**.
        - The Project has run before (the same ``project_name`` and ``project_name``), and (a) it stored models and / or
          artifacts, or (b) its status is Published , or (c) it is Archived.
        - A new Project is forced by calling ``Project.init`` with ``reuse_last_project_id=False``.

        Otherwise, the already initialized Project object for the same ``project_name`` and ``project_name`` is returned,
        or, when being executed remotely on a clearml-agent, the project returned is the existing project from the backend.

        .. note::
            To reference another Project, instead of initializing the same Project more than once, call
            :meth:`Project.get_project`. For example, to "share" the same experiment in more than one script,
            call ``Project.get_project``. See the ``Project.get_project`` method for an example.

        For example:
        The first time the following code runs, it will create a new Project. The status will be Completed.

        .. code-block:: py

            from clearml import Project
            project = Project.init('myProject', 'myProject')

        If this code runs again, it will not create a new Project. It does not store a model or artifact,
        it is not Published (its status Completed) , it was not Archived, and a new Project is not forced.

        If the Project is Published or Archived, and run again, it will create a new Project with a new Project ID.

        The following code will create a new Project every time it runs, because it stores an artifact.

        .. code-block:: py

            project = Project.init('myProject', 'myOtherProject')

            d = {'a': '1'}
            project.upload_artifact('myArtifact', d)

        :param str project_name: The name of the project in which the experiment will be created. If the project does
            not exist, it is created. If ``project_name`` is ``None``, the repository name is used. (Optional)
        :param str project_name: The name of Project (experiment). If ``project_name`` is ``None``, the Python experiment
            script's file name is used. (Optional)
        :param ProjectTypes project_type: The project type. Valid project types:

            - ``ProjectTypes.training`` (default)
            - ``ProjectTypes.testing``
            - ``ProjectTypes.inference``
            - ``ProjectTypes.data_processing``
            - ``ProjectTypes.application``
            - ``ProjectTypes.monitor``
            - ``ProjectTypes.controller``
            - ``ProjectTypes.optimizer``
            - ``ProjectTypes.service``
            - ``ProjectTypes.qc``
            - ``ProjectTypes.custom``

        :param tags: Add a list of tags (str) to the created Project. For example: tags=['512x512', 'yolov3']
        :param bool reuse_last_project_id: Force a new Project (experiment) with a previously used Project ID,
            and the same project and Project name. If the previously executed Project has artifacts or models, it will not be
            reused (overwritten), and a new Project will be created. When a Project is reused, the previous execution outputs
            are deleted, including console outputs and logs. The values are:

          - ``True`` - Reuse the last  Project ID. (default)
          - ``False`` - Force a new Project (experiment).
          - A string - You can also specify a Project ID (string) to be reused, instead of the cached ID based on the project/name combination.

        :param bool continue_last_project: Continue the execution of a previously executed Project (experiment). When
            continuing the executing of a previously executed Project,
            all previous artifacts / models / logs remain intact.
            New logs will continue iteration/step based on the previous-execution maximum iteration value.
            For example, The last train/loss scalar reported was iteration 100, the next report will be iteration 101.
            The values are:

          - ``True`` - Continue the last Project ID. Specified explicitly by reuse_last_project_id or implicitly with the same logic as reuse_last_project_id
          - ``False`` - Overwrite the execution of previous Project  (default).
          - A string - You can also specify a Project ID (string) to be continued. This is equivalent to `continue_last_project=True` and `reuse_last_project_id=a_project_id_string`.
          - An integer - Specify initial iteration offset (override the auto automatic last_iteration_offset). Pass 0, to disable the automatic last_iteration_offset or specify a different initial offset. You can specify a Project ID to be used with `reuse_last_project_id='project_id_here'`

        :param str output_uri: The default location for output models and other artifacts. If True, the default
            files_server will be used for model storage. In the default location, ClearML creates a subfolder for the
            output. If set to False, local runs will not upload output models and artifacts,
            and remote runs will not use any default values provided using ``default_output_uri``.
            The subfolder structure is the following: \<output destination name\> / \<project name\> / \<project name\>.\<Project ID\>.
            Note that for cloud storage, you must install the **ClearML** package for your cloud storage type,
            and then configure your storage credentials. For detailed information, see "Storage" in the ClearML
            Documentation.
            The following are examples of ``output_uri`` values for the supported locations:

          - A shared folder: ``/mnt/share/folder``
          - S3: ``s3://bucket/folder``
          - Google Cloud Storage: ``gs://bucket-name/folder``
          - Azure Storage: ``azure://company.blob.core.windows.net/folder/``
          - Default file server: True

        :param auto_connect_arg_parser: Automatically connect an argparse object to the Project. Supported argument
            parser packages are: argparse, click, python-fire, jsonargparse. The values are:

          - ``True`` - Automatically connect. (default)
          - ``False`` - Do not automatically connect.
          - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of connected
              arguments. The dictionary keys are argparse variable names and the values are booleans.
              The ``False`` value excludes the specified argument from the Project's parameter section.
              Keys missing from the dictionary default to ``True``, you can change it to be ``False`` by adding
              ``*`` key as ``False`` to the dictionary.
              An empty dictionary defaults to ``False``.

              For example:

              .. code-block:: py

                 auto_connect_arg_parser={"do_not_include_me": False, }

              .. code-block:: py

                 auto_connect_arg_parser={"only_include_me": True, "*": False}

              .. note::
               To manually connect an argparse, use :meth:`Project.connect`.

        :param auto_connect_frameworks: Automatically connect frameworks This includes patching MatplotLib, XGBoost,
            scikit-learn, Keras callbacks, and TensorBoard/X to serialize plots, graphs, and the model location to
            the **ClearML Server** (backend), in addition to original output destination.
            The values are:

          - ``True`` - Automatically connect (default)
          - ``False`` - Do not automatically connect
          - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of connected
              frameworks. The dictionary keys are frameworks and the values are booleans, other dictionaries used for
              finer control or wildcard strings.
              In case of wildcard strings, the local path of a model file has to match at least one wildcard to be
              saved/loaded by ClearML. Example: ``{'pytorch' : '*.pt', 'tensorflow': ['*.h5', '*']}``
              Keys missing from the dictionary default to ``True``, and an empty dictionary defaults to ``False``.
              Supported keys for finer control: ``{'tensorboard': {'report_hparams': bool}}``  # whether to report TensorBoard hyperparameters

              For example:

              .. code-block:: py

                 auto_connect_frameworks={
                     'matplotlib': True, 'tensorflow': ['*.hdf5, 'something_else*], 'tensorboard': True,
                     'pytorch': ['*.pt'], 'xgboost': True, 'scikit': True, 'fastai': True,
                     'lightgbm': True, 'hydra': True, 'detect_repository': True, 'tfdefines': True,
                     'joblib': True, 'megengine': True, 'catboost': True, 'gradio': True
                 }

              .. code-block:: py

                  auto_connect_frameworks={'tensorboard': {'report_hparams': False}}

        :param bool auto_resource_monitoring: Automatically create machine resource monitoring plots
            These plots appear in the **ClearML Web-App (UI)**, **RESULTS** tab, **SCALARS** sub-tab,
            with a title of **:resource monitor:**.
            The values are:

          - ``True`` - Automatically create resource monitoring plots. (default)
          - ``False`` - Do not automatically create.
          - Class Type - Create ResourceMonitor object of the specified class type.
          - dict - Dictionary of kwargs to be passed to the ResourceMonitor instance.
              The keys can be:
              - `report_start_sec` OR `first_report_sec` OR `seconds_from_start` - Maximum number of seconds
                  to wait for scalar/plot reporting before defaulting
                  to machine statistics reporting based on seconds from experiment start time
              - `wait_for_first_iteration_to_start_sec` - Set the initial time (seconds) to wait for iteration
                   reporting to be used as x-axis for the resource monitoring,
                   if timeout exceeds then reverts to `seconds_from_start`
              - `max_wait_for_first_iteration_to_start_sec` - Set the maximum time (seconds) to allow the resource
                  monitoring to revert back to iteration reporting x-axis after starting to report `seconds_from_start`
              - `report_mem_used_per_process` OR `report_global_mem_used` - Compatibility feature,
                  report memory usage for the entire machine
                  default (false), report only on the running process and its sub-processes

        :param auto_connect_streams: Control the automatic logging of stdout and stderr.
            The values are:

          - ``True`` - Automatically connect (default)
          -  ``False`` - Do not automatically connect
          - A dictionary - In addition to a boolean, you can use a dictionary for fined grained control of stdout and
              stderr. The dictionary keys are 'stdout' , 'stderr' and 'logging', the values are booleans.
              Keys missing from the dictionary default to ``False``, and an empty dictionary defaults to ``False``.
              Notice, the default behaviour is logging stdout/stderr. The `logging` module is logged as a by product
              of the stderr logging

              For example:

              .. code-block:: py

                 auto_connect_streams={'stdout': True, 'stderr': True, 'logging': False}

        :param deferred_init: (default: False) Wait for Project to be fully initialized (regular behaviour).
            ** BETA feature! use with care **.

            If set to True, `Project.init` function returns immediately and all initialization / communication
            to the clearml-server is running in a background thread. The returned object is
            a full proxy to the regular Project object, hence everything will be working as expected.
            Default behaviour can be controlled with: ``CLEARML_DEFERRED_TASK_INIT=1``. Notes:

          - Any access to the returned proxy `Project` object will essentially wait for the `Project.init` to be completed.
              For example: `print(project.name)` will wait for `Project.init` to complete in the
              background and then return the `name` property of the project original object
          - Before `Project.init` completes in the background, auto-magic logging (console/metric) might be missed
          - If running via an agent, this argument is ignored, and Project init is called synchronously (default)

        :return: The main execution Project (Project context)
        :rtype: Project
        """

        def verify_defaults_match():
            validate = [
                ('project name', project_name, cls.__main_project.get_project_name()),
                ('project name', project_name, cls.__main_project.name),
                ('project type', str(project_type) if project_type else project_type, str(cls.__main_project.project_type)),
            ]

            for field, default, current in validate:
                if default is not None and default != current:
                    raise UsageError(
                        "Current project already created "
                        "and requested {field} '{default}' does not match current {field} '{current}'. "
                        "If you wish to create additional projects use `Project.create`, "
                        "or close the current project with `project.close()` before calling `Project.init(...)`".format(
                            field=field,
                            default=default,
                            current=current,
                        )
                    )

        if cls.__main_project is not None and deferred_init != cls.__nested_deferred_init_flag:
            # if this is a subprocess, regardless of what the init was called for,
            # we have to fix the main project hooks and stdout bindings
            if cls.__forked_proc_main_pid != os.getpid() and cls.__is_subprocess():
                if project_type is None:
                    project_type = cls.__main_project.project_type
                # make sure we only do it once per process
                cls.__forked_proc_main_pid = os.getpid()
                # make sure we do not wait for the repo detect thread
                cls.__main_project._detect_repo_async_thread = None
                cls.__main_project._dev_worker = None
                cls.__main_project._resource_monitor = None

                # if we are using threads to send the reports,
                # after forking there are no threads, so we will need to recreate them
                if not getattr(cls, '_report_subprocess_enabled'):
                    # remove the logger from the previous process
                    cls.__main_project.get_logger()
                    # create a new logger (to catch stdout/err)
                    cls.__main_project._logger = None
                    cls.__main_project.__reporter = None
                    # noinspection PyProtectedMember
                    cls.__main_project._get_logger(auto_connect_streams=auto_connect_streams)
                    cls.__main_project._artifacts_manager = Artifacts(cls.__main_project)

                # unregister signal hooks, they cause subprocess to hang
                # noinspection PyProtectedMember
                cls.__main_project.__register_at_exit(cls.__main_project._at_exit)

                # if we are using threads to send the reports,
                # after forking there are no threads, so we will need to recreate them
                if not getattr(cls, '_report_subprocess_enabled'):
                    # start all reporting threads
                    BackgroundMonitor.start_all(project=cls.__main_project)

            if not running_remotely():
                verify_defaults_match()

            return cls.__main_project

        is_sub_process_project_id = None
        # check that we are not a child process, in that case do nothing.
        # we should not get here unless this is Windows/macOS platform, linux support fork
        if cls.__is_subprocess():
            is_sub_process_project_id = cls.__get_master_id_project_id()
            # we could not find a project ID, revert to old stub behaviour
            if not is_sub_process_project_id:
                return _ProjectStub()  # noqa

        elif running_remotely() and not get_is_master_node():
            # make sure we only do it once per process
            cls.__forked_proc_main_pid = os.getpid()
            # make sure everyone understands we should act as if we are a subprocess (fake pid 1)
            cls.__update_master_pid_project(pid=1, project=get_remote_project_id())
        else:
            # set us as master process (without project ID)
            cls.__update_master_pid_project()
            is_sub_process_project_id = None

        if project_type is None:
            # Backwards compatibility: if called from Project.current_project and project_type
            # was not specified, keep legacy default value of ProjectTypes.training
            project_type = cls.ProjectTypes.training
        elif isinstance(project_type, six.string_types):
            if project_type not in Project.ProjectTypes.__members__:
                raise ValueError("Project type '{}' not supported, options are: {}".format(
                    project_type, Project.ProjectTypes.__members__.keys()))
            project_type = Project.ProjectTypes.__members__[str(project_type)]

        is_deferred = False
        try:
            if not running_remotely():
                # check remote status
                _local_rank = get_torch_local_rank()
                if _local_rank is not None and _local_rank > 0:
                    is_sub_process_project_id = get_torch_distributed_anchor_project_id(timeout=30)

                # only allow if running locally and creating the first Project
                # otherwise we ignore and perform in order
                if ENV_DEFERRED_TASK_INIT.get():
                    deferred_init = True

                if not is_sub_process_project_id and deferred_init and deferred_init != cls.__nested_deferred_init_flag:
                    def completed_cb(x):
                        Project.__forked_proc_main_pid = os.getpid()
                        Project.__main_project = x

                    getLogger().warning("ClearML initializing Project in the background")

                    project = FutureProjectCaller(
                        func=cls.init,
                        func_cb=completed_cb,
                        override_cls=cls,
                        project_name=project_name,
                        project_name=project_name,
                        tags=tags,
                        reuse_last_project_id=reuse_last_project_id,
                        continue_last_project=continue_last_project,
                        output_uri=output_uri,
                        auto_connect_arg_parser=auto_connect_arg_parser,
                        auto_connect_frameworks=auto_connect_frameworks,
                        auto_resource_monitoring=auto_resource_monitoring,
                        auto_connect_streams=auto_connect_streams,
                        deferred_init=cls.__nested_deferred_init_flag,
                    )
                    is_deferred = True
                    # mark as temp master
                    cls.__update_master_pid_project()
                # if this is the main process, create the project
                elif not is_sub_process_project_id:
                    try:
                        project = cls._create_dev_project(
                            default_project_name=project_name,
                            default_project_name=project_name,
                            default_project_type=project_type,
                            tags=tags,
                            reuse_last_project_id=reuse_last_project_id,
                            continue_last_project=continue_last_project,
                            detect_repo=False if (
                                    isinstance(auto_connect_frameworks, dict) and
                                    not auto_connect_frameworks.get('detect_repository', True)) else True,
                            auto_connect_streams=auto_connect_streams,
                        )
                        # check if we are local rank 0 (local master),
                        # create an anchor with project ID for the other processes
                        if _local_rank == 0:
                            create_torch_distributed_anchor(project_id=project.id)

                    except MissingConfigError as e:
                        if not ENV_IGNORE_MISSING_CONFIG.get():
                            raise
                        getLogger().warning(str(e))
                        # return a Project-stub instead of the original class
                        # this will make sure users can still call the Stub without code breaking
                        return _ProjectStub()  # noqa
                    # set defaults
                    if cls._offline_mode:
                        project.output_uri = None
                        # create target data folder for logger / artifacts
                        # noinspection PyProtectedMember
                        Path(project._get_default_report_storage_uri()).mkdir(parents=True, exist_ok=True)
                    elif output_uri is not None:
                        if output_uri is True:
                            output_uri = project.get_project_object().default_output_destination or True
                        project.output_uri = output_uri
                    elif project.get_project_object().default_output_destination:
                        project.output_uri = project.get_project_object().default_output_destination
                    elif cls.__default_output_uri:
                        project.output_uri = str(cls.__default_output_uri)
                    # store new project ID
                    cls.__update_master_pid_project(project=project)
                else:
                    # subprocess should get back the project info
                    project = cls.get_project(project_id=is_sub_process_project_id)
            else:
                # if this is the main process, create the project
                if not is_sub_process_project_id:
                    project = cls(
                        private=cls.__create_protection,
                        project_id=get_remote_project_id(),
                        log_to_backend=False,
                    )
                    if output_uri is False and not project.output_uri:
                        # Setting output_uri=False argument will disable using any default when running remotely
                        pass
                    else:
                        if project.get_project_object().default_output_destination and not project.output_uri:
                            project.output_uri = project.get_project_object().default_output_destination
                        if cls.__default_output_uri and not project.output_uri:
                            project.output_uri = cls.__default_output_uri
                    # store new project ID
                    cls.__update_master_pid_project(project=project)
                    # make sure we are started
                    project.started(ignore_errors=True)
                    # continue last iteration if we had any (or we need to override it)
                    if isinstance(continue_last_project, int) and not isinstance(continue_last_project, bool):
                        project.set_initial_iteration(int(continue_last_project))
                    elif project.data.last_iteration:
                        project.set_initial_iteration(int(project.data.last_iteration) + 1)
                else:
                    # subprocess should get back the project info
                    project = cls.get_project(project_id=is_sub_process_project_id)
        except Exception:
            raise
        else:
            Project.__forked_proc_main_pid = os.getpid()
            Project.__main_project = project

            # register at exist only on the real (none deferred) Project
            if not is_deferred:
                # register the main project for at exit hooks (there should only be one)
                # noinspection PyProtectedMember
                project.__register_at_exit(project._at_exit)
                # noinspection PyProtectedMember
                if cls.__exit_hook:
                    cls.__exit_hook.register_signal_and_exception_hooks()

            # always patch OS forking because of ProcessPool and the alike
            PatchOsFork.patch_fork(project)
            if auto_connect_frameworks:
                def should_connect(*keys):
                    """
                    Evaluates value of auto_connect_frameworks[keys[0]]...[keys[-1]].
                    If at some point in the evaluation, the value of auto_connect_frameworks[keys[0]]...[keys[-1]]
                    is a bool, that value will be returned. If a dictionary is empty, it will be evaluated to False.
                    If a key will not be found in the current dictionary, True will be returned.
                    """
                    should_bind_framework = auto_connect_frameworks
                    for key in keys:
                        if not isinstance(should_bind_framework, dict):
                            return bool(should_bind_framework)
                        if should_bind_framework == {}:
                            return False
                        should_bind_framework = should_bind_framework.get(key, True)
                    return bool(should_bind_framework)

                if not is_deferred and should_connect("hydra"):
                    PatchHydra.update_current_project(project)
                if should_connect("scikit") and should_connect("joblib"):
                    PatchedJoblib.update_current_project(project)
                if should_connect("matplotlib"):
                    PatchedMatplotlib.update_current_project(project)
                if should_connect("tensorflow") or should_connect("tensorboard"):
                    # allow disabling tfdefines
                    if not is_deferred and should_connect("tfdefines"):
                        PatchAbsl.update_current_project(project)
                    TensorflowBinding.update_current_project(
                        project,
                        patch_reporting=should_connect("tensorboard"),
                        patch_model_io=should_connect("tensorflow"),
                        report_hparams=should_connect("tensorboard", "report_hparams"),
                    )
                if should_connect("pytorch"):
                    PatchPyTorchModelIO.update_current_project(project)
                if should_connect("megengine"):
                    PatchMegEngineModelIO.update_current_project(project)
                if should_connect("xgboost"):
                    PatchXGBoostModelIO.update_current_project(project)
                if should_connect("catboost"):
                    PatchCatBoostModelIO.update_current_project(project)
                if should_connect("fastai"):
                    PatchFastai.update_current_project(project)
                if should_connect("lightgbm"):
                    PatchLIGHTgbmModelIO.update_current_project(project)
                if should_connect("gradio"):
                    PatchGradio.update_current_project(project)

                cls.__add_model_wildcards(auto_connect_frameworks)

            # if we are deferred, stop here (the rest we do in the actual init)
            if is_deferred:
                from .backend_interface.logger import StdStreamPatch
                # patch console outputs, we will keep them in memory until we complete the Project init
                # notice we do not load config defaults, as they are not threadsafe
                # we might also need to override them with the vault
                StdStreamPatch.patch_std_streams(
                    project.get_logger(),
                    connect_stdout=(
                        auto_connect_streams is True) or (
                            isinstance(auto_connect_streams, dict) and auto_connect_streams.get('stdout', False)
                    ),
                    connect_stderr=(
                        auto_connect_streams is True) or (
                            isinstance(auto_connect_streams, dict) and auto_connect_streams.get('stderr', False)
                    ),
                    load_config_defaults=False,
                )
                return project  # noqa

            if auto_resource_monitoring and not is_sub_process_project_id:
                resource_monitor_cls = auto_resource_monitoring \
                    if isinstance(auto_resource_monitoring, six.class_types) else ResourceMonitor
                resource_monitor_kwargs = dict(
                    report_mem_used_per_process=not config.get("development.worker.report_global_mem_used", False),
                    first_report_sec=config.get("development.worker.report_start_sec", None),
                    wait_for_first_iteration_to_start_sec=config.get(
                        "development.worker.wait_for_first_iteration_to_start_sec", None
                    ),
                    max_wait_for_first_iteration_to_start_sec=config.get(
                        "development.worker.max_wait_for_first_iteration_to_start_sec", None
                    ),
                )
                if isinstance(auto_resource_monitoring, dict):
                    if "report_start_sec" in auto_resource_monitoring:
                        auto_resource_monitoring["first_report_sec"] = auto_resource_monitoring.pop("report_start_sec")
                    if "seconds_from_start" in auto_resource_monitoring:
                        auto_resource_monitoring["first_report_sec"] = auto_resource_monitoring.pop(
                            "seconds_from_start"
                        )
                    if "report_global_mem_used" in auto_resource_monitoring:
                        auto_resource_monitoring["report_mem_used_per_process"] = auto_resource_monitoring.pop(
                            "report_global_mem_used"
                        )
                    resource_monitor_kwargs.update(auto_resource_monitoring)
                project._resource_monitor = resource_monitor_cls(
                    project,
                    **resource_monitor_kwargs
                )
                project._resource_monitor.start()

            # make sure all random generators are initialized with new seed
            random_seed = project.get_random_seed()
            if random_seed is not None:
                make_deterministic(random_seed)
            project._set_random_seed_used(random_seed)

            if auto_connect_arg_parser:
                EnvironmentBind.update_current_project(project)

                PatchJsonArgParse.update_current_project(project)

                # Patch ArgParser to be aware of the current project
                argparser_update_currentproject(project)

                PatchClick.patch(project)
                PatchFire.patch(project)

                # set excluded arguments
                if isinstance(auto_connect_arg_parser, dict):
                    project._arguments.exclude_parser_args(auto_connect_arg_parser)

                # Check if parse args already called. If so, sync project parameters with parser
                if argparser_parseargs_called():
                    for parser, parsed_args in get_argparser_last_args():
                        project._connect_argparse(parser=parser, parsed_args=parsed_args)

                PatchHydra.delete_overrides()
            elif argparser_parseargs_called():
                # actually we have nothing to do, in remote running, the argparser will ignore
                # all non argparser parameters, only caveat if parameter connected with the same name
                # as the argparser this will be solved once sections are introduced to parameters
                pass

        # Make sure we start the logger, it will patch the main logging object and pipe all output
        # if we are running locally and using development mode worker, we will pipe all stdout to logger.
        # The logger will automatically take care of all patching (we just need to make sure to initialize it)
        logger = project._get_logger(auto_connect_streams=auto_connect_streams)
        # show the debug metrics page in the log, it is very convenient
        if not is_sub_process_project_id:
            if cls._offline_mode:
                logger.report_text('ClearML running in offline mode, session stored in {}'.format(
                    project.get_offline_mode_folder()))
            else:
                logger.report_text('ClearML results page: {}'.format(project.get_output_log_web_page()))
        # Make sure we start the dev worker if required, otherwise it will only be started when we write
        # something to the log.
        project._dev_mode_setup_worker()

        if (not project._reporter or not project._reporter.is_constructed()) and \
                is_sub_process_project_id and not cls._report_subprocess_enabled:
            project._setup_reporter()

        # start monitoring in background process or background threads
        # monitoring are: Resource monitoring and Dev Worker monitoring classes
        BackgroundMonitor.start_all(project=project)

        # noinspection PyProtectedMember
        project._set_startup_info()
        return project

    @classmethod
    def create(
            cls,
            project_name=None,  # type: Optional[str]
            project_name=None,  # type: Optional[str]
            project_type=None,  # type: Optional[str]
            repo=None,  # type: Optional[str]
            branch=None,  # type: Optional[str]
            commit=None,  # type: Optional[str]
            script=None,  # type: Optional[str]
            working_directory=None,  # type: Optional[str]
            packages=None,  # type: Optional[Union[bool, Sequence[str]]]
            requirements_file=None,  # type: Optional[Union[str, Path]]
            docker=None,  # type: Optional[str]
            docker_args=None,  # type: Optional[str]
            docker_bash_setup_script=None,  # type: Optional[str]
            argparse_args=None,  # type: Optional[Sequence[Tuple[str, str]]]
            base_project_id=None,  # type: Optional[str]
            add_project_init_call=True,  # type: bool
            force_single_script_file=False,  # type: bool
    ):
        # type: (...) -> ProjectInstance
        """
        Manually create and populate a new Project (experiment) in the system.
        If the code does not already contain a call to ``Project.init``, pass add_project_init_call=True,
        and the code will be patched in remote execution (i.e. when executed by `clearml-agent`)

        .. note::
           This method **always** creates a new Project.
           Use :meth:`Project.init` method to automatically create and populate project for the running process.
           To reference an existing Project, call the  :meth:`Project.get_project` method .

        :param project_name: Set the project name for the project. Required if base_project_id is None.
        :param project_name: Set the name of the remote project. Required if base_project_id is None.
        :param project_type: Optional, The project type to be created. Supported values: 'training', 'testing', 'inference',
            'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc', 'custom'
        :param repo: Remote URL for the repository to use, or path to local copy of the git repository
            Example: 'https://github.com/allegroai/clearml.git' or '~/project/repo'
        :param branch: Select specific repository branch/tag (implies the latest commit from the branch)
        :param commit: Select specific commit ID to use (default: latest commit,
            or when used with local repository matching the local commit id)
        :param script: Specify the entry point script for the remote execution. When used in tandem with
            remote git repository the script should be a relative path inside the repository,
            for example: './source/train.py' . When used with local repository path it supports a
            direct path to a file inside the local repository itself, for example: '~/project/source/train.py'
        :param working_directory: Working directory to launch the script from. Default: repository root folder.
            Relative to repo root or local folder.
        :param packages: Manually specify a list of required packages. Example: ``["tqdm>=2.1", "scikit-learn"]``
            or `True` to automatically create requirements
            based on locally installed packages (repository must be local).
        :param requirements_file: Specify requirements.txt file to install when setting the session.
            If not provided, the requirements.txt from the repository will be used.
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Project's environment
        :param argparse_args: Arguments to pass to the remote execution, list of string pairs (argument, value)
            Notice, only supported if the codebase itself uses argparse.ArgumentParser
        :param base_project_id: Use a pre-existing project in the system, instead of a local repo/script.
            Essentially clones an existing project and overrides arguments/requirements.
        :param add_project_init_call: If True, a 'Project.init()' call is added to the script entry point in remote execution.
        :param force_single_script_file: If True, do not auto-detect local repository

        :return: The newly created Project (experiment)
        :rtype: Project
        """
        if cls.is_offline():
            raise UsageError("Creating project in offline mode. Use 'Project.init' instead.")
        if not project_name and not base_project_id:
            if not cls.__main_project:
                raise ValueError("Please provide project_name, no global project context found "
                                 "(Project.current_project hasn't been called)")
            project_name = cls.__main_project.get_project_name()
        from .backend_interface.project.populate import CreateAndPopulate
        manual_populate = CreateAndPopulate(
            project_name=project_name, project_name=project_name, project_type=project_type,
            repo=repo, branch=branch, commit=commit,
            script=script, working_directory=working_directory,
            packages=packages, requirements_file=requirements_file,
            docker=docker, docker_args=docker_args, docker_bash_setup_script=docker_bash_setup_script,
            base_project_id=base_project_id,
            add_project_init_call=add_project_init_call,
            force_single_script_file=force_single_script_file,
            raise_on_missing_entries=False,
        )
        project = manual_populate.create_project()
        if project and argparse_args:
            manual_populate.update_project_args(argparse_args)
            project.reload()

        return project

    @classmethod
    def get_by_name(cls, project_name):
        # type: (str) -> ProjectInstance
        """

        .. note::
            This method is deprecated, use :meth:`Project.get_project` instead.

        Returns the most recent project with the given name from anywhere in the system as a Project object.

        :param str project_name: The name of the project to search for.

        :return: Project object of the most recent project with that name.
        """
        warnings.warn("Warning: 'Project.get_by_name' is deprecated. Use 'Project.get_project' instead", DeprecationWarning)
        return cls.get_project(project_name=project_name)

    @classmethod
    def get_project(
            cls,
            project_id=None,  # type: Optional[str]
            project_name=None,  # type: Optional[str]
            project_name=None,  # type: Optional[str]
            tags=None,  # type: Optional[Sequence[str]]
            allow_archived=True,  # type: bool
            project_filter=None  # type: Optional[dict]
    ):
        # type: (...) -> ProjectInstance
        """
        Get a Project by ID, or project name / project name combination.

        For example:

        The following code demonstrates calling ``Project.get_project`` to report a scalar to another Project. The output
        of :meth:`.Logger.report_scalar` from testing is associated with the Project named ``training``. It allows
        training and testing to run concurrently, because they initialized different Projects (see :meth:`Project.init`
        for information about initializing Projects).

        The training script:

        .. code-block:: py

            # initialize the training Project
            project = Project.init('myProject', 'training')

            # do some training

        The testing script:

        .. code-block:: py

            # initialize the testing Project
            project = Project.init('myProject', 'testing')

            # get the training Project
            train_project = Project.get_project(project_name='myProject', project_name='training')

            # report metrics in the training Project
            for x in range(10):
                train_project.get_logger().report_scalar('title', 'series', value=x * 2, iteration=x)

        :param str project_id: The ID (system UUID) of the experiment to get.
            If specified, ``project_name`` and ``project_name`` are ignored.
        :param str project_name: The project name of the Project to get.
        :param str project_name: The name of the Project within ``project_name`` to get.
        :param list tags: Filter based on the requested list of tags (strings). To exclude a tag add "-" prefix to the
            tag. Example: ``["best", "-debug"]``.
            The default behaviour is to join all tags with a logical "OR" operator.
            To join all tags with a logical "AND" operator instead, use "__$all" as the first string, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever"]

            To join all tags with AND, but exclude a tag use "__$not" before the excluded tag, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever", "__$not", "internal", "__$not", "test"]

            The "OR" and "AND" operators apply to all tags that follow them until another operator is specified.
            The NOT operator applies only to the immediately following tag.
            For example:

            .. code-block:: py

                ["__$all", "a", "b", "c", "__$or", "d", "__$not", "e", "__$and", "__$or" "f", "g"]

            This example means ("a" AND "b" AND "c" AND ("d" OR NOT "e") AND ("f" OR "g")).
            See https://clear.ml/docs/latest/docs/clearml_sdk/project_sdk/#tag-filters for more information.
        :param bool allow_archived: Only applicable if *not* using specific ``project_id``,
            If True (default), allow to return archived Projects, if False filter out archived Projects
        :param bool project_filter: Only applicable if *not* using specific ``project_id``,
            Pass additional query filters, on top of project/name. See details in Project.get_projects.

        :return: The Project specified by ID, or project name / experiment name combination.
        :rtype: Project
        """
        return cls.__get_project(
            project_id=project_id, project_name=project_name, project_name=project_name, tags=tags,
            include_archived=allow_archived, project_filter=project_filter,
        )

    @classmethod
    def get_projects(
            cls,
            project_ids=None,  # type: Optional[Sequence[str]]
            project_name=None,  # type: Optional[Union[Sequence[str],str]]
            project_name=None,  # type: Optional[str]
            tags=None,  # type: Optional[Sequence[str]]
            allow_archived=True,  # type: bool
            project_filter=None  # type: Optional[Dict]
    ):
        # type: (...) -> List[ProjectInstance]
        """
        Get a list of Projects objects matching the queries/filters

        - A list of specific Project IDs.
        - Filter Projects based on specific fields:
            project name (including partial match), project name (including partial match), tags
            Apply Additional advanced filtering with `project_filter`

        .. note::
            This function returns the most recent 500 projects. If you wish to retrieve older projects
            use ``Project.query_projects()``

        :param list(str) project_ids: The IDs (system UUID) of experiments to get.
            If ``project_ids`` specified, then ``project_name`` and ``project_name`` are ignored.
        :param str project_name: The project name of the Projects to get. To get the experiment
            in all projects, use the default value of ``None``. (Optional)
            Use a list of strings for multiple optional project names.
        :param str project_name: The full name or partial name of the Projects to match within the specified
            ``project_name`` (or all projects if ``project_name`` is ``None``).
            This method supports regular expressions for name matching (if you wish to match special characters and
            avoid any regex behaviour, use re.escape()). (Optional)
            To match an exact project name (i.e. not partial matching),
            add ^/$ at the beginning/end of the string, for example: "^exact_project_name_here$"
        :param list tags: Filter based on the requested list of tags (strings). To exclude a tag add "-" prefix to the
            tag. Example: ``["best", "-debug"]``.
            The default behaviour is to join all tags with a logical "OR" operator.
            To join all tags with a logical "AND" operator instead, use "__$all" as the first string, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever"]

            To join all tags with AND, but exclude a tag use "__$not" before the excluded tag, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever", "__$not", "internal", "__$not", "test"]

            The "OR" and "AND" operators apply to all tags that follow them until another operator is specified.
            The NOT operator applies only to the immediately following tag.
            For example:

            .. code-block:: py

                ["__$all", "a", "b", "c", "__$or", "d", "__$not", "e", "__$and", "__$or" "f", "g"]

            This example means ("a" AND "b" AND "c" AND ("d" OR NOT "e") AND ("f" OR "g")).
            See https://clear.ml/docs/latest/docs/clearml_sdk/project_sdk/#tag-filters for more information.
        :param bool allow_archived: If True (default), allow to return archived Projects, if False filter out archived Projects
        :param dict project_filter: filter and order Projects.
            See :class:`.backend_api.service.v?.projects.GetAllRequest` for details; the ? needs to be replaced by the appropriate version.

          - ``parent`` - (str) filter by parent project-id matching
          - ``search_text`` - (str) free text search (in project fields comment/name/id)
          - ``status`` - List[str] List of valid statuses. Options are: "created", "queued", "in_progress", "stopped", "published", "publishing", "closed", "failed", "completed", "unknown"
          - ``type`` - List[str] List of valid project types. Options are: 'training', 'testing', 'inference', 'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc'. 'custom'
          - ``user`` - List[str] Filter based on Project's user owner, provide list of valid user IDs.
          - ``order_by`` - List[str] List of field names to order by. When ``search_text`` is used. Use '-' prefix to specify descending order. Optional, recommended when using page. Example: ``order_by=['-last_update']``
          - ``_all_`` - dict(fields=[], pattern='')  Match string `pattern` (regular expression) appearing in All `fields`. Example: dict(fields=['script.repository'], pattern='github.com/user')
          - ``_any_`` - dict(fields=[], pattern='')  Match string `pattern` (regular expression) appearing in Any of the `fields`. Example: dict(fields=['comment', 'name'], pattern='my comment')
          - Examples - ``{'status': ['stopped'], 'order_by': ["-last_update"]}`` , ``{'order_by'=['-last_update'], '_all_'=dict(fields=['script.repository'], pattern='github.com/user'))``

        :return: The Projects specified by the parameter combinations (see the parameters).
        :rtype: List[Project]
        """
        project_filter = project_filter or {}
        if not allow_archived:
            project_filter['system_tags'] = (project_filter.get('system_tags') or []) + ['-{}'.format(cls.archived_tag)]

        return cls.__get_projects(project_ids=project_ids, project_name=project_name, tags=tags,
                               project_name=project_name, **project_filter)

    @classmethod
    def query_projects(
            cls,
            project_name=None,  # type: Optional[Union[Sequence[str],str]]
            project_name=None,  # type: Optional[str]
            tags=None,  # type: Optional[Sequence[str]]
            additional_return_fields=None,  # type: Optional[Sequence[str]]
            project_filter=None,  # type: Optional[Dict]
    ):
        # type: (...) -> Union[List[str], List[Dict[str, str]]]
        """
        Get a list of Projects ID matching the specific query/filter.
        Notice, if `additional_return_fields` is specified, returns a list of
        dictionaries with requested fields (dict per Project)

        :param str project_name: The project name of the Projects to get. To get the experiment
            in all projects, use the default value of ``None``. (Optional)
            Use a list of strings for multiple optional project names.
        :param str project_name: The full name or partial name of the Projects to match within the specified
            ``project_name`` (or all projects if ``project_name`` is ``None``).
            This method supports regular expressions for name matching (if you wish to match special characters and
            avoid any regex behaviour, use re.escape()). (Optional)
        :param str project_name: project name (str) the project belongs to (use None for all projects)
        :param str project_name: project name (str) within the selected project
            Return any partial match of project_name, regular expressions matching is also supported.
            If None is passed, returns all projects within the project
        :param list tags: Filter based on the requested list of tags (strings).
            To exclude a tag add "-" prefix to the tag. Example: ``["best", "-debug"]``.
            The default behaviour is to join all tags with a logical "OR" operator.
            To join all tags with a logical "AND" operator instead, use "__$all" as the first string, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever"]

            To join all tags with AND, but exclude a tag use "__$not" before the excluded tag, for example:

            .. code-block:: py

                ["__$all", "best", "experiment", "ever", "__$not", "internal", "__$not", "test"]

            The "OR" and "AND" operators apply to all tags that follow them until another operator is specified.
            The NOT operator applies only to the immediately following tag.
            For example:

            .. code-block:: py

                ["__$all", "a", "b", "c", "__$or", "d", "__$not", "e", "__$and", "__$or" "f", "g"]

            This example means ("a" AND "b" AND "c" AND ("d" OR NOT "e") AND ("f" OR "g")).
            See https://clear.ml/docs/latest/docs/clearml_sdk/project_sdk/#tag-filters for more information.
        :param list additional_return_fields: Optional, if not provided return a list of Project IDs.
            If provided return dict per Project with the additional requested fields.
            Example: ``returned_fields=['last_updated', 'user', 'script.repository']`` will return a list of dict:
            ``[{'id': 'project_id', 'last_update': datetime.datetime(), 'user': 'user_id', 'script.repository': 'https://github.com/user/'}, ]``
        :param dict project_filter: filter and order Projects.
            See :class:`.backend_api.service.v?.projects.GetAllRequest` for details; the ? needs to be replaced by the appropriate version.

          - ``parent`` - (str) filter by parent project-id matching
          - ``search_text`` - (str) free text search (in project fields comment/name/id)
          - ``status`` - List[str] List of valid statuses. Options are: "created", "queued", "in_progress", "stopped", "published", "publishing", "closed", "failed", "completed", "unknown"
          - ``type`` - List[Union[str, ProjectTypes]] List of valid project types. Options are: 'training', 'testing', 'inference', 'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc'. 'custom'
          - ``user`` - List[str] Filter based on Project's user owner, provide list of valid user IDs.
          - ``order_by`` - List[str] List of field names to order by. When search_text is used. Use '-' prefix to specify descending order. Optional, recommended when using page. Example: ``order_by=['-last_update']``
          - ``_all_`` - dict(fields=[], pattern='')  Match string ``pattern`` (regular expression) appearing in All `fields`. ``dict(fields=['script.repository'], pattern='github.com/user')``
          - ``_any_`` - dict(fields=[], pattern='')  Match string `pattern` (regular expression) appearing in Any of the `fields`. `dict(fields=['comment', 'name'], pattern='my comment')`
          - Examples: ``{'status': ['stopped'], 'order_by': ["-last_update"]}``, ``{'order_by'=['-last_update'], '_all_'=dict(fields=['script.repository'], pattern='github.com/user')}``

        :return: The Projects specified by the parameter combinations (see the parameters).
        """
        project_filter = project_filter or {}
        if tags:
            project_filter['tags'] = (project_filter.get('tags') or []) + list(tags)
        return_fields = {}
        if additional_return_fields:
            return_fields = set(list(additional_return_fields) + ['id'])
            project_filter['only_fields'] = (project_filter.get('only_fields') or []) + list(return_fields)

        if project_filter.get('type'):
            project_filter['type'] = [str(project_type) for project_type in project_filter['type']]

        results = cls._query_projects(project_name=project_name, project_name=project_name, **project_filter)
        return [t.id for t in results] if not additional_return_fields else \
            [{k: cls._get_data_property(prop_path=k, data=r, raise_on_error=False, log_on_error=False)
              for k in return_fields}
             for r in results]

    @property
    def output_uri(self):
        # type: () -> str
        """
        The storage / output url for this project. This is the default location for output models and other artifacts.

        :return: The url string.
        """
        return self.storage_uri

    @property
    def last_worker(self):
        # type: () -> str
        """
        ID of last worker that handled the project.

        :return: The worker ID.
        """
        return self._data.last_worker

    @output_uri.setter
    def output_uri(self, value):
        # type: (Union[str, bool]) -> None
        """
        Set the storage / output url for this project. This is the default location for output models and other artifacts.

        :param str/bool value: The value to set for output URI. Can be either a bucket link, True for default server
            or False. Check Project.init reference docs for more info (output_uri is a parameter).
        """

        # check if this is boolean
        if value is False:
            value = None
        elif value is True:
            value = str(self.__default_output_uri or self._get_default_report_storage_uri())

        # check if we have the correct packages / configuration
        if value and value != self.storage_uri:
            from .storage.helper import StorageHelper
            helper = StorageHelper.get(value)
            if not helper:
                raise ValueError("Could not get access credentials for '{}' "
                                 ", check configuration file ~/clearml.conf".format(value))
            helper.check_write_permissions(value)
        self.storage_uri = value

    @property
    def artifacts(self):
        # type: () -> Dict[str, Artifact]
        """
        A read-only dictionary of Project artifacts (name, artifact).

        :return: The artifacts.
        """
        if not Session.check_min_api_version('2.3'):
            return ReadOnlyDict()
        artifacts_pairs = []
        if self.data.execution and self.data.execution.artifacts:
            artifacts_pairs = [(a.key, Artifact(a)) for a in self.data.execution.artifacts]
        if self._artifacts_manager:
            artifacts_pairs += list(self._artifacts_manager.registered_artifacts.items())
        return ReadOnlyDict(artifacts_pairs)

    @property
    def models(self):
        # type: () -> Mapping[str, Sequence[Model]]
        """
        Read-only dictionary of the Project's loaded/stored models.

        :return: A dictionary-like object with "input"/"output" keys and input/output properties, pointing to a
            list-like object containing Model objects. Each list-like object also acts as a dictionary, mapping
            model name to an appropriate model instance.

            Get input/output models:

            .. code-block:: py

                project.models.input
                project.models["input"]

                project.models.output
                project.models["output"]

            Get the last output model:

            .. code-block:: py

                project.models.output[-1]

            Get a model by name:

            .. code-block:: py

                project.models.output["model name"]
        """
        return self.get_models()

    @property
    def logger(self):
        # type: () -> Logger
        """
        Get a Logger object for reporting, for this project context. You can view all Logger report output associated with
        the Project for which this method is called, including metrics, plots, text, tables, and images, in the
        **ClearML Web-App (UI)**.

        :return: The Logger object for the current Project (experiment).
        """
        return self.get_logger()

    @classmethod
    def clone(
            cls,
            source_project=None,  # type: Optional[Union[Project, str]]
            name=None,  # type: Optional[str]
            comment=None,  # type: Optional[str]
            parent=None,  # type: Optional[str]
            project=None,  # type: Optional[str]
    ):
        # type: (...) -> ProjectInstance
        """
        Create a duplicate (a clone) of a Project (experiment). The status of the cloned Project is ``Draft``
        and modifiable.

        Use this method to manage experiments and for autoML.

        :param str source_project: The Project to clone. Specify a Project object or a  Project ID. (Optional)
        :param str name: The name of the new cloned Project. (Optional)
        :param str comment: A comment / description for the new cloned Project. (Optional)
        :param str parent: The ID of the parent Project of the new Project.

          - If ``parent`` is not specified, then ``parent`` is set to ``source_project.parent``.
          - If ``parent`` is not specified and ``source_project.parent`` is not available, then ``parent`` set to ``source_project``.

        :param str project: The ID of the project in which to create the new Project.
            If ``None``, the new project inherits the original Project's project. (Optional)

        :return: The new cloned Project (experiment).
        :rtype: Project
        """
        assert isinstance(source_project, (six.string_types, Project))
        if not Session.check_min_api_version('2.4'):
            raise ValueError("ClearML-server does not support DevOps features, "
                             "upgrade clearml-server to 0.12.0 or above")

        project_id = source_project if isinstance(source_project, six.string_types) else source_project.id
        if not parent:
            if isinstance(source_project, six.string_types):
                source_project = cls.get_project(project_id=source_project)
            parent = source_project.id if not source_project.parent else source_project.parent
        elif isinstance(parent, Project):
            parent = parent.id

        cloned_project_id = cls._clone_project(cloned_project_id=project_id, name=name, comment=comment,
                                         parent=parent, project=project)
        cloned_project = cls.get_project(project_id=cloned_project_id)
        return cloned_project

    @classmethod
    def enqueue(cls, project, queue_name=None, queue_id=None, force=False):
        # type: (Union[Project, str], Optional[str], Optional[str], bool) -> Any
        """
        Enqueue a Project for execution, by adding it to an execution queue.

        .. note::
           A worker daemon must be listening at the queue for the worker to fetch the Project and execute it,
           see "ClearML Agent" in the ClearML Documentation.

        :param Project/str project: The Project to enqueue. Specify a Project object or  Project ID.
        :param str queue_name: The name of the queue. If not specified, then ``queue_id`` must be specified.
        :param str queue_id: The ID of the queue. If not specified, then ``queue_name`` must be specified.
        :param bool force: If True, reset the Project if necessary before enqueuing it

        :return: An enqueue JSON response.

            .. code-block:: javascript

               {
                    "queued": 1,
                    "updated": 1,
                    "fields": {
                        "status": "queued",
                        "status_reason": "",
                        "status_message": "",
                        "status_changed": "2020-02-24T15:05:35.426770+00:00",
                        "last_update": "2020-02-24T15:05:35.426770+00:00",
                        "execution.queue": "2bd96ab2d9e54b578cc2fb195e52c7cf"
                        }
                }

            - ``queued``  - The number of Projects enqueued (an integer or ``null``).
            - ``updated`` - The number of Projects updated (an integer or ``null``).
            - ``fields``

              - ``status`` - The status of the experiment.
              - ``status_reason`` - The reason for the last status change.
              - ``status_message`` - Information about the status.
              - ``status_changed`` - The last status change date and time (ISO 8601 format).
              - ``last_update`` - The last Project update time, including Project creation, update, change, or events for this project (ISO 8601 format).
              - ``execution.queue`` - The ID of the queue where the Project is enqueued. ``null`` indicates not enqueued.

        """
        assert isinstance(project, (six.string_types, Project))
        if not Session.check_min_api_version('2.4'):
            raise ValueError("ClearML-server does not support DevOps features, "
                             "upgrade clearml-server to 0.12.0 or above")

        # make sure we have wither name ot id
        mutually_exclusive(queue_name=queue_name, queue_id=queue_id)

        project_id = project if isinstance(project, six.string_types) else project.id
        session = cls._get_default_session()
        if not queue_id:
            queue_id = get_queue_id(session, queue_name)
            if not queue_id:
                raise ValueError('Could not find queue named "{}"'.format(queue_name))

        req = projects.EnqueueRequest(project=project_id, queue=queue_id)
        exception = None
        res = None
        try:
            res = cls._send(session=session, req=req)
            ok = res.ok()
        except Exception as e:
            exception = e
            ok = False
        if not ok:
            if not force:
                if res:
                    raise ValueError(res.response)
                raise exception
            project = cls.get_project(project_id=project) if isinstance(project, str) else project
            project.reset(set_started_on_success=False, force=True)
            req = projects.EnqueueRequest(project=project_id, queue=queue_id)
            res = cls._send(session=session, req=req)
            if not res.ok():
                raise ValueError(res.response)
        resp = res.response
        return resp

    @classmethod
    def get_num_enqueued_projects(cls, queue_name=None, queue_id=None):
        # type: (Optional[str], Optional[str]) -> int
        """
        Get the number of projects enqueued in a given queue.

        :param queue_name: The name of the queue. If not specified, then ``queue_id`` must be specified
        :param queue_id: The ID of the queue. If not specified, then ``queue_name`` must be specified

        :return: The number of projects enqueued in the given queue
        """
        if not Session.check_min_api_server_version("2.20", raise_error=True):
            raise ValueError("You version of clearml-server does not support the 'queues.get_num_entries' endpoint")
        mutually_exclusive(queue_name=queue_name, queue_id=queue_id)
        session = cls._get_default_session()
        if not queue_id:
            queue_id = get_queue_id(session, queue_name)
            if not queue_id:
                raise ValueError('Could not find queue named "{}"'.format(queue_name))
        result = get_num_enqueued_projects(session, queue_id)
        if result is None:
            raise ValueError("Could not query the number of enqueued projects in queue with ID {}".format(queue_id))
        return result

    @classmethod
    def dequeue(cls, project):
        # type: (Union[Project, str]) -> Any
        """
        Dequeue (remove) a Project from an execution queue.

        :param Project/str project: The Project to dequeue. Specify a Project object or  Project ID.

        :return: A dequeue JSON response.

        .. code-block:: javascript

           {
                "dequeued": 1,
                "updated": 1,
                "fields": {
                    "status": "created",
                    "status_reason": "",
                    "status_message": "",
                    "status_changed": "2020-02-24T16:43:43.057320+00:00",
                    "last_update": "2020-02-24T16:43:43.057320+00:00",
                    "execution.queue": null
                    }
            }

        - ``dequeued``  - The number of Projects enqueued (an integer or ``null``).
        - ``fields``

          - ``status`` - The status of the experiment.
          - ``status_reason`` - The reason for the last status change.
          - ``status_message`` - Information about the status.
          - ``status_changed`` - The last status change date and time in ISO 8601 format.
          - ``last_update`` - The last time the Project was created, updated,
                changed, or events for this project were reported.
          - ``execution.queue`` - The ID of the queue where the Project is enqueued. ``null`` indicates not enqueued.

        - ``updated`` - The number of Projects updated (an integer or ``null``).

        """
        assert isinstance(project, (six.string_types, Project))
        if not Session.check_min_api_version('2.4'):
            raise ValueError("ClearML-server does not support DevOps features, "
                             "upgrade clearml-server to 0.12.0 or above")

        project_id = project if isinstance(project, six.string_types) else project.id
        session = cls._get_default_session()
        req = projects.DequeueRequest(project=project_id)
        res = cls._send(session=session, req=req)
        resp = res.response
        return resp

    def set_progress(self, progress):
        # type: (int) -> ()
        """
        Sets Project's progress (0 - 100)
        Progress is a field computed and reported by the user.

        :param progress: numeric value (0 - 100)
        """
        if not isinstance(progress, int) or progress < 0 or progress > 100:
            self.log.warning("Can't set progress {} as it is not and int between 0 and 100".format(progress))
            return
        self._set_runtime_properties({"progress": str(progress)})

    def get_progress(self):
        # type: () -> (Optional[int])
        """
        Gets Project's progress (0 - 100)

        :return: Project's progress as an int.
            In case the progress doesn't exist, None will be returned
        """
        progress = self._get_runtime_properties().get("progress")
        if progress is None or not progress.isnumeric():
            return None
        return int(progress)

    def add_tags(self, tags):
        # type: (Union[Sequence[str], str]) -> None
        """
        Add Tags to this project. Old tags are not deleted. When executing a Project (experiment) remotely,
        this method has no effect.

        :param tags: A list of tags which describe the Project to add.
        """

        if isinstance(tags, six.string_types):
            tags = tags.split(" ")

        self.data.tags = list(set((self.data.tags or []) + tags))
        self._edit(tags=self.data.tags)

    def connect(self, mutable, name=None, ignore_remote_overrides=False):
        # type: (Any, Optional[str], bool) -> Any
        """
        Connect an object to a Project object. This connects an experiment component (part of an experiment) to the
        experiment. For example, an experiment component can be a valid object containing some hyperparameters, or a :class:`Model`.
        When running remotely, the value of the connected object is overridden by the corresponding value found
        under the experiment's UI/backend (unless `ignore_remote_overrides` is True).

        :param object mutable: The experiment component to connect. The object must be one of the following types:

          - argparse - An argparse object for parameters.
          - dict - A dictionary for parameters. Note: only keys of type `str` are supported.
          - ProjectParameters - A ProjectParameters object.
          - :class:`Model` - A model object for initial model warmup, or for model update/snapshot uploading. In practice the model should be either :class:`InputModel` or :class:`OutputModel`.
          - type - A Class type, storing all class properties (excluding '_' prefixed properties).
          - object - A class instance, storing all instance properties (excluding '_' prefixed properties).

        :param str name: A section name associated with the connected object, if 'name' is None defaults to 'General'
            Currently, `name` is only supported for `dict` and `ProjectParameter` objects, and should be omitted for the other supported types. (Optional)
            For example, by setting `name='General'` the connected dictionary will be under the General section in the hyperparameters section.
            While by setting `name='Train'` the connected dictionary will be under the Train section in the hyperparameters section.

        :param ignore_remote_overrides: If True, ignore UI/backend overrides when running remotely.
        Default is False, meaning that any changes made in the UI/backend will be applied in remote execution.

        :return: It will return the same object that was passed as the `mutable` argument to the method, except if the type of the object is dict.
                 For dicts the :meth:`Project.connect` will return the dict decorated as a `ProxyDictPostWrite`.
                 This is done to allow propagating the updates from the connected object.

        :raise: Raises an exception if passed an unsupported object.
        """
        # input model connect and project parameters will handle this instead
        if not isinstance(mutable, (InputModel, ProjectParameters)):
            ignore_remote_overrides = self._handle_ignore_remote_overrides(
                (name or "General") + "/_ignore_remote_overrides_", ignore_remote_overrides
            )
        # dispatching by match order
        dispatch = (
            (OutputModel, self._connect_output_model),
            (InputModel, self._connect_input_model),
            (ArgumentParser, self._connect_argparse),
            (dict, self._connect_dictionary),
            (ProjectParameters, self._connect_project_parameters),
            (type, self._connect_object),
            (object, self._connect_object),
        )

        multi_config_support = Session.check_min_api_version('2.9')
        if multi_config_support and not name and not isinstance(mutable, (OutputModel, InputModel)):
            name = self._default_configuration_section_name

        if not multi_config_support and name and name != self._default_configuration_section_name:
            raise ValueError("Multiple configurations is not supported with the current 'clearml-server', "
                             "please upgrade to the latest version")

        for mutable_type, method in dispatch:
            if isinstance(mutable, mutable_type):
                return method(mutable, name=name, ignore_remote_overrides=ignore_remote_overrides)

        raise Exception('Unsupported mutable type %s: no connect function found' % type(mutable).__name__)

    def set_packages(self, packages):
        # type: (Union[str, Path, Sequence[str]]) -> ()
        """
        Manually specify a list of required packages or a local requirements.txt file. Note that this will
        overwrite all existing packages.

        When running remotely this call is ignored

        :param packages: The list of packages or the path to the requirements.txt file.

            Example: ``["tqdm>=2.1", "scikit-learn"]`` or ``"./requirements.txt"`` or ``""``
            Use an empty string (packages="") to clear the requirements section (remote execution will use
                requirements.txt from the git repository if the file exists)
        """
        if running_remotely() or packages is None:
            return
        self._wait_for_repo_detection(timeout=300.)

        if packages and isinstance(packages, (str, Path)) and Path(packages).is_file():
            with open(Path(packages).as_posix(), "rt") as f:
                # noinspection PyProtectedMember
                self._update_requirements([line.strip() for line in f.readlines()])
            return

        # noinspection PyProtectedMember
        self._update_requirements(packages or "")

    def set_repo(self, repo=None, branch=None, commit=None):
        # type: (Optional[str], Optional[str], Optional[str]) -> ()
        """
        Specify a repository to attach to the function.
        Allow users to execute the project inside the specified repository, enabling them to load modules/script
        from the repository. Notice the execution work directory will be the repository root folder.
        Supports both git repo url link, and local repository path (automatically converted into the remote
        git/commit as is currently checkout).
        Example remote url: "https://github.com/user/repo.git".
        Example local repo copy: "./repo" - will automatically store the remote
        repo url and commit ID based on the locally cloned copy.
        When executing remotely, this call will not override the repository data (it is ignored)

        :param repo: Optional, remote URL for the repository to use, OR path to local copy of the git repository.
            Use an empty string to clear the repo.
            Example: "https://github.com/allegroai/clearml.git" or "~/project/repo" or ""
        :param branch: Optional, specify the remote repository branch (Ignored, if local repo path is used).
            Use an empty string to clear the branch.
        :param commit: Optional, specify the repository commit ID (Ignored, if local repo path is used).
            Use an empty string to clear the commit.
        """
        if running_remotely():
            return
        self._wait_for_repo_detection(timeout=300.)
        with self._edit_lock:
            self.reload()
            if repo is not None:
                # we cannot have None on the value itself
                self.data.script.repository = repo or ""
            if branch is not None:
                # we cannot have None on the value itself
                self.data.script.branch = branch or ""
            if commit is not None:
                # we cannot have None on the value itself
                self.data.script.version_num = commit or ""
            self._edit(script=self.data.script)

    def get_requirements(self):
        # type: () -> RequirementsDict
        """
        Get the project's requirements

        :return: A `RequirementsDict` object that holds the `pip`, `conda`, `orig_pip` requirements.
        """
        if not running_remotely() and self.is_main_project():
            self._wait_for_repo_detection(timeout=300.)
        requirements_dict = RequirementsDict()
        requirements_dict.update(self.data.script.requirements)
        return requirements_dict

    def connect_configuration(self, configuration, name=None, description=None, ignore_remote_overrides=False):
        # type: (Union[Mapping, list, Path, str], Optional[str], Optional[str], bool) -> Union[dict, Path, str]
        """
        Connect a configuration dictionary or configuration file (pathlib.Path / str) to a Project object.
        This method should be called before reading the configuration file.

        For example, a local file:

        .. code-block:: py

           config_file = project.connect_configuration(config_file)
           my_params = json.load(open(config_file,'rt'))

        A parameter dictionary/list:

        .. code-block:: py

           my_params = project.connect_configuration(my_params)

        When running remotely, the value of the connected configuration is overridden by the corresponding value found
        under the experiment's UI/backend (unless `ignore_remote_overrides` is True).

        :param configuration: The configuration. This is usually the configuration used in the model training process.
            Specify one of the following:

          - A dictionary/list - A dictionary containing the configuration. ClearML stores the configuration in
              the **ClearML Server** (backend), in a HOCON format (JSON-like format) which is editable.
          - A ``pathlib2.Path`` string - A path to the configuration file. ClearML stores the content of the file.
              A local path must be relative path. When executing a Project remotely in a worker, the contents brought
              from the **ClearML Server** (backend) overwrites the contents of the file.

        :param str name: Configuration section name. default: 'General'
            Allowing users to store multiple configuration dicts/files

        :param str description: Configuration section description (text). default: None

        :param bool ignore_remote_overrides: If True, ignore UI/backend overrides when running remotely.
        Default is False, meaning that any changes made in the UI/backend will be applied in remote execution.

        :return: If a dictionary is specified, then a dictionary is returned. If pathlib2.Path / string is
            specified, then a path to a local configuration file is returned. Configuration object.
        """
        ignore_remote_overrides = self._handle_ignore_remote_overrides(
            (name or "General") + "/_ignore_remote_overrides_config_", ignore_remote_overrides
        )
        pathlib_Path = None  # noqa
        cast_Path = Path
        if not isinstance(configuration, (dict, list, Path, six.string_types)):
            try:
                from pathlib import Path as pathlib_Path  # noqa
            except ImportError:
                pass
            if not pathlib_Path or not isinstance(configuration, pathlib_Path):
                raise ValueError("connect_configuration supports `dict`, `str` and 'Path' types, "
                                 "{} is not supported".format(type(configuration)))
        if pathlib_Path and isinstance(configuration, pathlib_Path):
            cast_Path = pathlib_Path

        multi_config_support = Session.check_min_api_version('2.9')
        if multi_config_support and not name:
            name = self._default_configuration_section_name

        if not multi_config_support and name and name != self._default_configuration_section_name:
            raise ValueError("Multiple configurations is not supported with the current 'clearml-server', "
                             "please upgrade to the latest version")

        # parameter dictionary
        if isinstance(configuration, (dict, list,)):
            def _update_config_dict(project, config_dict):
                if multi_config_support:
                    # noinspection PyProtectedMember
                    project._set_configuration(
                        name=name, description=description, config_type='dictionary', config_dict=config_dict)
                else:
                    # noinspection PyProtectedMember
                    project._set_model_config(config_dict=config_dict)

            def get_dev_config(configuration_):
                if multi_config_support:
                    self._set_configuration(
                        name=name, description=description, config_type="dictionary", config_dict=configuration_
                    )
                else:
                    self._set_model_config(config_dict=configuration)
                if isinstance(configuration_, dict):
                    configuration_ = ProxyDictPostWrite(self, _update_config_dict, configuration_)
                return configuration_

            if not running_remotely() or not (self.is_main_project() or self._is_remote_main_project()) or ignore_remote_overrides:
                configuration = get_dev_config(configuration)
            else:
                # noinspection PyBroadException
                try:
                    remote_configuration = self._get_configuration_dict(name=name) \
                        if multi_config_support else self._get_model_config_dict()
                except Exception:
                    remote_configuration = None

                if remote_configuration is None:
                    LoggerRoot.get_base_logger().warning(
                        "Could not retrieve remote configuration named \'{}\'\n"
                        "Using default configuration: {}".format(name, str(configuration)))
                    # update back configuration section
                    if multi_config_support:
                        self._set_configuration(
                            name=name, description=description,
                            config_type='dictionary', config_dict=configuration)
                    return configuration

                if not remote_configuration:
                    configuration = get_dev_config(configuration)
                elif isinstance(configuration, dict):
                    configuration.clear()
                    configuration.update(remote_configuration)
                    configuration = ProxyDictPreWrite(False, False, **configuration)
                elif isinstance(configuration, list):
                    configuration.clear()
                    configuration.extend(remote_configuration)

            return configuration

        # it is a path to a local file
        if not running_remotely() or not (self.is_main_project() or self._is_remote_main_project()) or ignore_remote_overrides:
            # check if not absolute path
            configuration_path = cast_Path(configuration)
            if not configuration_path.is_file():
                ValueError("Configuration file does not exist")
            try:
                with open(configuration_path.as_posix(), 'rt') as f:
                    configuration_text = f.read()
            except Exception:
                raise ValueError("Could not connect configuration file {}, file could not be read".format(
                    configuration_path.as_posix()))
            if multi_config_support:
                self._set_configuration(
                    name=name, description=description,
                    config_type=configuration_path.suffixes[-1].lstrip('.')
                    if configuration_path.suffixes and configuration_path.suffixes[-1] else 'file',
                    config_text=configuration_text)
            else:
                self._set_model_config(config_text=configuration_text)
            return configuration
        else:
            configuration_text = self._get_configuration_text(name=name) if multi_config_support \
                else self._get_model_config_text()
            if configuration_text is None:
                LoggerRoot.get_base_logger().warning(
                    "Could not retrieve remote configuration named \'{}\'\n"
                    "Using default configuration: {}".format(name, str(configuration)))
                # update back configuration section
                if multi_config_support:
                    configuration_path = cast_Path(configuration)
                    if configuration_path.is_file():
                        with open(configuration_path.as_posix(), 'rt') as f:
                            configuration_text = f.read()

                        self._set_configuration(
                            name=name, description=description,
                            config_type=configuration_path.suffixes[-1].lstrip('.')
                            if configuration_path.suffixes and configuration_path.suffixes[-1] else 'file',
                            config_text=configuration_text)
                return configuration

            configuration_path = cast_Path(configuration)
            fd, local_filename = mkstemp(prefix='clearml_project_config_',
                                         suffix=configuration_path.suffixes[-1] if
                                         configuration_path.suffixes else '.txt')
            with open(fd, "w") as f:
                f.write(configuration_text)
            return cast_Path(local_filename) if isinstance(configuration, cast_Path) else local_filename

    def connect_label_enumeration(self, enumeration, ignore_remote_overrides=False):
        # type: (Dict[str, int], bool) -> Dict[str, int]
        """
        Connect a label enumeration dictionary to a Project (experiment) object.

        Later, when creating an output model, the model will include the label enumeration dictionary.

        :param dict enumeration: A label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }

        :param ignore_remote_overrides: If True, ignore UI/backend overrides when running remotely.
            Default is False, meaning that any changes made in the UI/backend will be applied in remote execution.
        :return: The label enumeration dictionary (JSON).
        """
        ignore_remote_overrides = self._handle_ignore_remote_overrides(
            "General/_ignore_remote_overrides_label_enumeration_", ignore_remote_overrides
        )
        if not isinstance(enumeration, dict):
            raise ValueError("connect_label_enumeration supports only `dict` type, "
                             "{} is not supported".format(type(enumeration)))

        if (
            not running_remotely()
            or not (self.is_main_project() or self._is_remote_main_project())
            or ignore_remote_overrides
        ):
            self.set_model_label_enumeration(enumeration)
        else:
            # pop everything
            enumeration.clear()
            enumeration.update(self.get_labels_enumeration())
        return enumeration

    def get_logger(self):
        # type: () -> Logger
        """
        Get a Logger object for reporting, for this project context. You can view all Logger report output associated with
        the Project for which this method is called, including metrics, plots, text, tables, and images, in the
        **ClearML Web-App (UI)**.

        :return: The Logger for the Project (experiment).
        """
        return self._get_logger(auto_connect_streams=self._log_to_backend)

    def launch_multi_node(
        self,
        total_num_nodes,  # type: int
        port=29500,  # type: Optional[int]
        queue=None,  # type: Optional[str]
        wait=False,  # type: bool
        addr=None,  # type: Optional[str]
        devices=None,  # type: Optional[Union[int, Sequence[int]]]
        hide_children=False  # bool
    ):
        """
        Enqueue multiple clones of the current project to a queue, allowing the project
        to be ran by multiple workers in parallel. Each project running this way is called a node.
        Each node has a rank The node that initialized the execution of the other nodes
        is called the `master node` and it has a rank equal to 0.

        A dictionary named `multi_node_instance` will be connected to the projects.
        One can use this dictionary to modify the behaviour of this function when running remotely.
        The contents of this dictionary correspond to the parameters of this function, and they are:
        - `total_num_nodes` - the total number of nodes, including the master node
        - `queue` - the queue to enqueue the nodes to

        The following environment variables, will be set:
        - `MASTER_ADDR` - the address of the machine that the master node is running on
        - `MASTER_PORT` - the open port of the machine that the master node is running on
        - `WORLD_SIZE` - the total number of nodes, including the master
        - `RANK` - the rank of the current node (master has rank 0)

        One may use this function in conjuction with PyTorch's distributed communication package.
        Note that `Project.launch_multi_node` should be called before `torch.distributed.init_process_group`.
        For example:

        .. code-block:: py

            from clearml import Project
            import torch
            import torch.distributed as dist

            def run(rank, size):
                print('World size is ', size)
                tensor = torch.zeros(1)
                if rank == 0:
                    for i in range(1, size):
                        tensor += 1
                        dist.send(tensor=tensor, dst=i)
                        print('Sending from rank ', rank, ' to rank ', i, ' data: ', tensor[0])
                else:
                    dist.recv(tensor=tensor, src=0)
                    print('Rank ', rank, ' received data: ', tensor[0])

            if __name__ == '__main__':
                project = Project.init('some_name', 'some_name')
                project.execute_remotely(queue_name='queue')
                config = project.launch_multi_node(4)
                dist.init_process_group('gloo')
                run(config.get('node_rank'), config.get('total_num_nodes'))

        When using the ClearML cloud autoscaler apps, one needs to make sure the nodes can reach eachother.
        The machines need to be in the same security group, the `MASTER_PORT` needs to be exposed and the
        `MASTER_ADDR` needs to be the right private ip of the instance the master is running on.
        For example, to achieve this, one can set the following Docker arguments in the `Additional ClearML Configuration` section:

        .. code-block:: py

            agent.extra_docker_arguments=["--ipc=host", "--network=host", "-p", "29500:29500", "--env", "CLEARML_MULTI_NODE_MASTER_DEF_ADDR=`hostname -I | awk '{print $1}'`"]`

        :param total_num_nodes: The total number of nodes to be enqueued, including the master node,
            which should already be enqueued when running remotely
        :param port: Port opened by the master node. If the environment variable ``CLEARML_MULTI_NODE_MASTER_DEF_PORT``
            is set, the value of this parameter will be set to the one defined in ``CLEARML_MULTI_NODE_MASTER_DEF_PORT``.
            If ``CLEARML_MULTI_NODE_MASTER_DEF_PORT`` doesn't exist, but ``MASTER_PORT`` does, then the value of this
            parameter will be set to the one defined in ``MASTER_PORT``. If neither environment variables exist,
            the value passed to the parameter will be used
        :param queue: The queue to enqueue the nodes to. Can be different from the queue the master
            node is enqueued to. If None, the nodes will be enqueued to the same queue as the master node
        :param wait: If True, the master node will wait for the other nodes to start
        :param addr: The address of the master node's worker. If the environment variable
            ``CLEARML_MULTI_NODE_MASTER_DEF_ADDR`` is set, the value of this parameter will be set to
            the one defined in ``CLEARML_MULTI_NODE_MASTER_DEF_ADDR``.
            If ``CLEARML_MULTI_NODE_MASTER_DEF_ADDR`` doesn't exist, but ``MASTER_ADDR`` does, then the value of this
            parameter will be set to the one defined in ``MASTER_ADDR``. If neither environment variables exist,
            the value passed to the parameter will be used. If this value is None (default), the private IP of
            the machine the master node is running on will be used.
        :param devices: The devices to use. This can be a positive number indicating the number of devices to use,
            a sequence of indices or the value ``-1`` to indicate all available devices should be used.
        :param hide_children: If True, the children projects will be hidden. Otherwise, they will be visible in the UI

        :return: A dictionary containing relevant information regarding the multi node run. This dictionary has the following entries:

          - `master_addr` - the address of the machine that the master node is running on
          - `master_port` - the open port of the machine that the master node is running on
          - `total_num_nodes` - the total number of nodes, including the master
          - `queue` - the queue the nodes are enqueued to, excluding the master
          - `node_rank` - the rank of the current node (master has rank 0)
          - `wait` - if True, the master node will wait for the other nodes to start
        """

        def set_launch_multi_node_runtime_props(project, conf):
            # noinspection PyProtectedMember
            project._set_runtime_properties(
                {"{}/{}".format(self._launch_multi_node_section, k): v for k, v in conf.items()}
            )

        if total_num_nodes < 1:
            raise UsageError("total_num_nodes needs to be at least 1")
        if running_remotely() and not (self.data.execution and self.data.execution.queue) and not queue:
            raise UsageError("Master project is not enqueued to any queue and the queue parameter is None")

        master_conf = {
            "master_addr": os.environ.get(
                "CLEARML_MULTI_NODE_MASTER_DEF_ADDR", os.environ.get("MASTER_ADDR", addr or get_private_ip())
            ),
            "master_port": int(
                os.environ.get("CLEARML_MULTI_NODE_MASTER_DEF_PORT", os.environ.get("MASTER_PORT", port))
            ),
            "node_rank": 0,
            "wait": wait,
            "devices": devices
        }
        editable_conf = {"total_num_nodes": total_num_nodes, "queue": queue}
        editable_conf = self.connect(editable_conf, name=self._launch_multi_node_section)
        if not running_remotely():
            return master_conf
        master_conf.update(editable_conf)
        runtime_properties = self._get_runtime_properties()
        remote_node_rank = runtime_properties.get("{}/node_rank".format(self._launch_multi_node_section))

        current_conf = master_conf
        if remote_node_rank:
            # self is a child node, build the conf from the runtime proprerties
            current_conf = {
                entry: runtime_properties.get("{}/{}".format(self._launch_multi_node_section, entry))
                for entry in master_conf.keys()
            }
        elif os.environ.get("CLEARML_MULTI_NODE_MASTER") is None:
            nodes_to_wait = []
            # self is the master node, enqueue the other nodes
            set_launch_multi_node_runtime_props(self, master_conf)
            for node_rank in range(1, master_conf.get("total_num_nodes", total_num_nodes)):
                node = self.clone(source_project=self, parent=self.id)
                node_conf = copy.deepcopy(master_conf)
                node_conf["node_rank"] = node_rank
                set_launch_multi_node_runtime_props(node, node_conf)
                node.set_system_tags(
                    node.get_system_tags()
                    + [self._launch_multi_node_instance_tag]
                    + ([self.__hidden_tag] if hide_children else [])
                )
                if master_conf.get("queue"):
                    Project.enqueue(node, queue_name=master_conf["queue"])
                else:
                    Project.enqueue(node, queue_id=self.data.execution.queue)
                if master_conf.get("wait"):
                    nodes_to_wait.append(node)
            for node_to_wait, rank in zip(nodes_to_wait, range(1, master_conf.get("total_num_nodes", total_num_nodes))):
                self.log.info("Waiting for node with project ID {} and rank {}".format(node_to_wait.id, rank))
                node_to_wait.wait_for_status(
                    status=(
                        Project.ProjectStatusEnum.completed,
                        Project.ProjectStatusEnum.stopped,
                        Project.ProjectStatusEnum.closed,
                        Project.ProjectStatusEnum.failed,
                        Project.ProjectStatusEnum.in_progress,
                    ),
                    check_interval_sec=10,
                )
                self.log.info("Node with project ID {} and rank {} detected".format(node_to_wait.id, rank))
            os.environ["CLEARML_MULTI_NODE_MASTER"] = "1"

        num_devices = 1
        if devices is not None:
            try:
                num_devices = int(devices)
            except TypeError:
                try:
                    num_devices = len(devices)
                except Exception as ex:
                    raise ValueError("Failed parsing number of devices: {}".format(ex))
            except ValueError as ex:
                raise ValueError("Failed parsing number of devices: {}".format(ex))
            if num_devices < 0:
                try:
                    import torch

                    num_devices = torch.cuda.device_count()
                except ImportError:
                    raise ImportError(
                        "Could not import `torch` while finding the number of devices. "
                        "Please install it or set `devices` to a value different than -1"
                    )

        os.environ["MASTER_ADDR"] = current_conf.get("master_addr", "")
        os.environ["MASTER_PORT"] = str(current_conf.get("master_port", ""))
        os.environ["RANK"] = str(
            current_conf.get("node_rank", 0) * num_devices + int(os.environ.get("LOCAL_RANK", "0"))
        )
        os.environ["NODE_RANK"] = str(current_conf.get("node_rank", ""))
        os.environ["WORLD_SIZE"] = str(current_conf.get("total_num_nodes", total_num_nodes) * num_devices)

        return current_conf

    def mark_started(self, force=False):
        # type: (bool) -> ()
        """
        Manually mark a Project as started (happens automatically)

        :param bool force: If True, the project status will be changed to `started` regardless of the current Project state.
        """
        # UI won't let us see metrics if we're not started
        self.started(force=force)
        self.reload()

    def mark_stopped(self, force=False, status_message=None):
        # type: (bool, Optional[str]) -> ()
        """
        Manually mark a Project as stopped (also used in :meth:`_at_exit`)

        :param bool force: If True, the project status will be changed to `stopped` regardless of the current Project state.
        :param str status_message: Optional, add status change message to the stop request.
            This message will be stored as status_message on the Project's info panel
        """
        # flush any outstanding logs
        self.flush(wait_for_uploads=True)
        # mark project as stopped
        self.stopped(force=force, status_message=str(status_message) if status_message else None)

    def flush(self, wait_for_uploads=False):
        # type: (bool) -> bool
        """
        Flush any outstanding reports or console logs.

        :param bool wait_for_uploads: Wait for all outstanding uploads to complete

            - ``True`` - Wait
            - ``False`` - Do not wait (default)
        """

        # make sure model upload is done
        if BackendModel.get_num_results() > 0 and wait_for_uploads:
            BackendModel.wait_for_results()

        # flush any outstanding logs
        if self._logger:
            # noinspection PyProtectedMember
            self._logger._flush_stdout_handler()
        if self.__reporter:
            self.__reporter.flush()
            if wait_for_uploads:
                self.__reporter.wait_for_events()

        LoggerRoot.flush()

        return True

    def reset(self, set_started_on_success=False, force=False):
        # type: (bool, bool) -> None
        """
        Reset a Project. ClearML reloads a Project after a successful reset.
        When a worker executes a Project remotely, the Project does not reset unless
        the ``force`` parameter is set to ``True`` (this avoids accidentally clearing logs and metrics).

        :param bool set_started_on_success: If successful, automatically set the Project to `started`

            - ``True`` - If successful, set to started.
            - ``False`` - If successful, do not set to started. (default)

        :param bool force: Force a Project reset, even when executing the Project (experiment) remotely in a worker

            - ``True`` - Force
            - ``False`` - Do not force (default)
        """
        if not running_remotely() or not self.is_main_project() or force:
            super(Project, self).reset(set_started_on_success=set_started_on_success, force=force)

    def close(self):
        """
        Closes the current Project and changes its status to "Completed".
        Enables you to manually shut down the project from the process which opened the project.

        This method does not terminate the (current) Python process, in contrast to :meth:`Project.mark_completed`.

        After having :meth:`Project.close` -d a project, the respective object cannot be used anymore and
        methods like :meth:`Project.connect` or :meth:`Project.connect_configuration` will throw a `ValueError`.
        In order to obtain an object representing the project again, use methods like :meth:`Project.get_project`.

        .. warning::
           Only call :meth:`Project.close` if you are certain the Project is not needed.
        """
        if self._at_exit_called:
            return

        # store is main before we call at_exit, because will will Null it
        is_main = self.is_main_project()
        is_sub_process = self.__is_subprocess()

        # wait for repository detection (5 minutes should be reasonable time to detect all packages)
        if self._logger and not self.__is_subprocess():
            self._wait_for_repo_detection(timeout=300.)

        self.__shutdown()
        # unregister atexit callbacks and signal hooks, if we are the main project
        if is_main:
            self.__register_at_exit(None)
            self._remove_signal_hooks()
            self._remove_exception_hooks()
            if not is_sub_process:
                # make sure we enable multiple Project.init callas with reporting sub-processes
                BackgroundMonitor.clear_main_process(self)
                # noinspection PyProtectedMember
                Logger._remove_std_logger()

                # unbind everything
                PatchHydra.update_current_project(None)
                PatchedJoblib.update_current_project(None)
                PatchedMatplotlib.update_current_project(None)
                PatchAbsl.update_current_project(None)
                TensorflowBinding.update_current_project(None)
                PatchPyTorchModelIO.update_current_project(None)
                PatchMegEngineModelIO.update_current_project(None)
                PatchXGBoostModelIO.update_current_project(None)
                PatchCatBoostModelIO.update_current_project(None)
                PatchFastai.update_current_project(None)
                PatchLIGHTgbmModelIO.update_current_project(None)
                EnvironmentBind.update_current_project(None)
                PatchJsonArgParse.update_current_project(None)
                PatchOsFork.patch_fork(None)

    def delete(
            self,
            delete_artifacts_and_models=True,
            skip_models_used_by_other_projects=True,
            raise_on_error=False,
            callback=None,
    ):
        # type: (bool, bool, bool, Callable[[str, str], bool]) -> bool
        """
        Delete the project as well as its output models and artifacts.
        Models and artifacts are deleted from their storage locations, each using its URI.

        Note: in order to delete models and artifacts using their URI, make sure the proper storage credentials are
        configured in your configuration file (e.g. if an artifact is stored in S3, make sure sdk.aws.s3.credentials
        are properly configured and that you have delete permission in the related buckets).

        :param delete_artifacts_and_models: If True, artifacts and models would also be deleted (default True).
                                            If callback is provided, this argument is ignored.
        :param skip_models_used_by_other_projects: If True, models used by other projects would not be deleted (default True)
        :param raise_on_error: If True, an exception will be raised when encountering an error.
                               If False an error would be printed and no exception will be raised.
        :param callback: An optional callback accepting a uri type (string) and a uri (string) that will be called
                         for each artifact and model. If provided, the delete_artifacts_and_models is ignored.
                         Return True to indicate the artifact/model should be deleted or False otherwise.
        :return: True if the project was deleted successfully.
        """
        if not running_remotely() or not self.is_main_project():
            return super(Project, self)._delete(
                delete_artifacts_and_models=delete_artifacts_and_models,
                skip_models_used_by_other_projects=skip_models_used_by_other_projects,
                raise_on_error=raise_on_error,
                callback=callback,
            )
        return False

    def register_artifact(self, name, artifact, metadata=None, uniqueness_columns=True):
        # type: (str, pandas.DataFrame, Dict, Union[bool, Sequence[str]]) -> None
        """
        Register (add) an artifact for the current Project. Registered artifacts are dynamically synchronized with the
        **ClearML Server** (backend). If a registered artifact is updated, the update is stored in the
        **ClearML Server** (backend). Registered artifacts are primarily used for Data Auditing.

        The currently supported registered artifact object type is a pandas.DataFrame.

        See also :meth:`Project.unregister_artifact` and :meth:`Project.get_registered_artifacts`.

        .. note::
           ClearML also supports uploaded artifacts which are one-time uploads of static artifacts that are not
           dynamically synchronized with the **ClearML Server** (backend). These static artifacts include
           additional object types. For more information, see :meth:`Project.upload_artifact`.

        :param str name: The name of the artifact.

         .. warning::
            If an artifact with the same name was previously registered, it is overwritten.
        :param object artifact: The artifact object.
        :param dict metadata: A dictionary of key-value pairs for any metadata. This dictionary appears with the
            experiment in the **ClearML Web-App (UI)**, **ARTIFACTS** tab.
        :param uniqueness_columns: A Sequence of columns for artifact uniqueness comparison criteria, or the default
            value of ``True``. If ``True``, the artifact uniqueness comparison criteria is all the columns,
            which is the same as ``artifact.columns``.
        """
        if not isinstance(uniqueness_columns, CollectionsSequence) and uniqueness_columns is not True:
            raise ValueError('uniqueness_columns should be a List (sequence) or True')
        if isinstance(uniqueness_columns, str):
            uniqueness_columns = [uniqueness_columns]
        self._artifacts_manager.register_artifact(
            name=name, artifact=artifact, metadata=metadata, uniqueness_columns=uniqueness_columns)

    def unregister_artifact(self, name):
        # type: (str) -> None
        """
        Unregister (remove) a registered artifact. This removes the artifact from the watch list that ClearML uses
        to synchronize artifacts with the **ClearML Server** (backend).

        .. important::
           - Calling this method does not remove the artifact from a Project. It only stops ClearML from
             monitoring the artifact.
           - When this method is called, ClearML immediately takes the last snapshot of the artifact.
        """
        self._artifacts_manager.unregister_artifact(name=name)

    def get_registered_artifacts(self):
        # type: () -> Dict[str, Artifact]
        """
        Get a dictionary containing the Project's registered (dynamically synchronized) artifacts (name, artifact object).

        .. note::
           After calling ``get_registered_artifacts``, you can still modify the registered artifacts.

        :return: The registered (dynamically synchronized) artifacts.
        """
        return self._artifacts_manager.registered_artifacts

    def upload_artifact(
            self,
            name,  # type: str
            artifact_object,  # type: Union[str, Mapping, pandas.DataFrame, numpy.ndarray, Image.Image, Any]
            metadata=None,  # type: Optional[Mapping]
            delete_after_upload=False,  # type: bool
            auto_pickle=True,  # type: bool
            preview=None,  # type: Any
            wait_on_upload=False,  # type: bool
            extension_name=None,  # type: Optional[str]
            serialization_function=None,  # type: Optional[Callable[[Any], Union[bytes, bytearray]]]
            retries=0  # type: int
    ):
        # type: (...) -> bool
        """
        Upload (add) a static artifact to a Project object. The artifact is uploaded in the background.

        The currently supported upload (static) artifact types include:

        - string / pathlib2.Path - A path to artifact file. If a wildcard or a folder is specified, then ClearML
          creates and uploads a ZIP file.
        - dict - ClearML stores a dictionary as ``.json`` (or see ``extension_name``) file and uploads it.
        - pandas.DataFrame - ClearML stores a pandas.DataFrame as ``.csv.gz`` (compressed CSV)
          (or see ``extension_name``) file and uploads it.
        - numpy.ndarray - ClearML stores a numpy.ndarray as ``.npz`` (or see ``extension_name``) file and uploads it.
        - PIL.Image - ClearML stores a PIL.Image as ``.png`` (or see ``extension_name``) file and uploads it.
        - Any - If called with auto_pickle=True, the object will be pickled and uploaded.

        :param str name: The artifact name.

            .. warning::
               If an artifact with the same name was previously uploaded, then it is overwritten.

        :param object artifact_object:  The artifact object.
        :param dict metadata: A dictionary of key-value pairs for any metadata. This dictionary appears with the
            experiment in the **ClearML Web-App (UI)**, **ARTIFACTS** tab.
        :param bool delete_after_upload: After the upload, delete the local copy of the artifact

            - ``True`` - Delete the local copy of the artifact.
            - ``False`` - Do not delete. (default)

        :param bool auto_pickle: If True (default) and the artifact_object is not one of the following types:
            pathlib2.Path, dict, pandas.DataFrame, numpy.ndarray, PIL.Image, url (string), local_file (string),
            the artifact_object will be pickled and uploaded as pickle file artifact (with file extension .pkl)

        :param Any preview: The artifact preview

        :param bool wait_on_upload: Whether the upload should be synchronous, forcing the upload to complete
            before continuing.

        :param str extension_name: File extension which indicates the format the artifact should be stored as.
            The following are supported, depending on the artifact type (default value applies when extension_name is None):

          - Any - ``.pkl`` if passed supersedes any other serialization type, and always pickles the object
          - dict - ``.json``, ``.yaml`` (default ``.json``)
          - pandas.DataFrame - ``.csv.gz``, ``.parquet``, ``.feather``, ``.pickle`` (default ``.csv.gz``)
          - numpy.ndarray - ``.npz``, ``.csv.gz`` (default ``.npz``)
          - PIL.Image - whatever extensions PIL supports (default ``.png``)
          - In case the ``serialization_function`` argument is set - any extension is supported

        :param Callable[Any, Union[bytes, bytearray]] serialization_function: A serialization function that takes one
            parameter of any type which is the object to be serialized. The function should return
            a `bytes` or `bytearray` object, which represents the serialized object. Note that the object will be
            immediately serialized using this function, thus other serialization methods will not be used
            (e.g. `pandas.DataFrame.to_csv`), even if possible. To deserialize this artifact when getting
            it using the `Artifact.get` method, use its `deserialization_function` argument.

        :param int retries: Number of retries before failing to upload artifact. If 0, the upload is not retried

        :return: The status of the upload.

            - ``True`` - Upload succeeded.
            - ``False`` - Upload failed.

        :raise: If the artifact object type is not supported, raise a ``ValueError``.
        """
        exception_to_raise = None
        for retry in range(retries + 1):
            # noinspection PyBroadException
            try:
                if self._artifacts_manager.upload_artifact(
                    name=name,
                    artifact_object=artifact_object,
                    metadata=metadata,
                    delete_after_upload=delete_after_upload,
                    auto_pickle=auto_pickle,
                    preview=preview,
                    wait_on_upload=wait_on_upload,
                    extension_name=extension_name,
                    serialization_function=serialization_function,
                ):
                    return True
            except Exception as e:
                exception_to_raise = e
            if retry < retries:
                getLogger().warning(
                    "Failed uploading artifact '{}'. Retrying... ({}/{})".format(name, retry + 1, retries)
                )
        if exception_to_raise:
            raise exception_to_raise
        return False

    def get_debug_samples(self, title, series, n_last_iterations=None):
        # type: (str, str, Optional[int]) -> List[dict]
        """
        :param str title: Debug sample's title, also called metric in the UI
        :param str series: Debug sample's series,
            corresponding to debug sample's file name in the UI, also known as variant
        :param int n_last_iterations: How many debug sample iterations to fetch in reverse chronological order.
            Leave empty to get all debug samples.

        :raise: TypeError if `n_last_iterations` is explicitly set to anything other than a positive integer value

        :return: A list of `dict`s, each dictionary containing the debug sample's URL and other metadata.
            The URLs can be passed to StorageManager.get_local_copy to fetch local copies of debug samples.
        """
        from .config.defs import MAX_SERIES_PER_METRIC

        if not n_last_iterations:
            n_last_iterations = MAX_SERIES_PER_METRIC.get()

        if isinstance(n_last_iterations, int) and n_last_iterations >= 0:
            samples = self._get_debug_samples(
                title=title, series=series, n_last_iterations=n_last_iterations
            )
        else:
            raise TypeError(
                "Parameter n_last_iterations is expected to be a positive integer value,"
                " but instead got n_last_iterations={}".format(n_last_iterations)
            )

        return samples

    def _send_debug_image_request(self, title, series, n_last_iterations, scroll_id=None):
        return Project._send(
            Project._get_default_session(),
            events.DebugImagesRequest(
                [{"project": self.id, "metric": title, "variants": [series]}],
                iters=n_last_iterations,
                scroll_id=scroll_id,
            ),
        )

    def _get_debug_samples(self, title, series, n_last_iterations=None):
        response = self._send_debug_image_request(title, series, n_last_iterations)

        debug_samples = []

        while True:
            scroll_id = response.response_data.get("scroll_id", None)

            for metric_resp in response.response_data.get("metrics", []):
                iterations_events = [iteration["events"] for iteration in metric_resp.get("iterations", [])]  # type: List[List[dict]]
                flattened_events = (event
                                    for single_iter_events in iterations_events
                                    for event in single_iter_events)
                debug_samples.extend(flattened_events)

            response = self._send_debug_image_request(
                title, series, n_last_iterations, scroll_id=scroll_id
            )

            if (len(debug_samples) == n_last_iterations
                or all(
                    len(metric_resp.get("iterations", [])) == 0
                    for metric_resp in response.response_data.get("metrics", []))):
                break

        return debug_samples

    def get_models(self):
        # type: () -> Mapping[str, Sequence[Model]]
        """
        Return a dictionary with ``{'input': [], 'output': []}`` loaded/stored models of the current Project
        Input models are files loaded in the project, either manually or automatically logged
        Output models are files stored in the project, either manually or automatically logged.
        Automatically logged frameworks are for example: TensorFlow, Keras, PyTorch, ScikitLearn(joblib) etc.

        :return: A dictionary-like object with "input"/"output" keys and input/output properties, pointing to a
            list-like object containing Model objects. Each list-like object also acts as a dictionary, mapping
            model name to an appropriate model instance.

            Example:

            .. code-block:: py

                {'input': [clearml.Model()], 'output': [clearml.Model()]}

        """
        return ProjectModels(self)

    def is_current_project(self):
        # type: () -> bool
        """
        .. deprecated:: 0.13.0
           This method is deprecated. Use :meth:`Project.is_main_project` instead.

        Is this Project object the main execution Project (initially returned by :meth:`Project.init`)

        :return: Is this Project object the main execution Project

            - ``True`` - Is the main execution Project.
            - ``False`` - Is not the main execution Project.

        """
        return self.is_main_project()

    def is_main_project(self):
        # type: () -> bool
        """
        Is this Project object the main execution Project (initially returned by :meth:`Project.init`)

        .. note::
           If :meth:`Project.init` was never called, this method will *not* create
           it, making this test more efficient than:

           .. code-block:: py

              Project.init() == project

        :return: Is this Project object the main execution Project

            - ``True`` - Is the main execution Project.
            - ``False`` - Is not the main execution Project.

        """
        return self is self.__main_project

    def set_model_config(self, config_text=None, config_dict=None):
        # type: (Optional[str], Optional[Mapping]) -> None
        """
        .. deprecated:: 0.14.1
            Use :meth:`Project.connect_configuration` instead.
        """
        self._set_model_config(config_text=config_text, config_dict=config_dict)

    def get_model_config_text(self):
        # type: () -> str
        """
        .. deprecated:: 0.14.1
            Use :meth:`Project.connect_configuration` instead.
        """
        return self._get_model_config_text()

    def get_model_config_dict(self):
        # type: () -> Dict
        """
        .. deprecated:: 0.14.1
            Use :meth:`Project.connect_configuration` instead.
        """
        return self._get_model_config_dict()

    def set_model_label_enumeration(self, enumeration=None):
        # type: (Optional[Mapping[str, int]]) -> ()
        """
        Set the label enumeration for the Project object before creating an output model.
        Later, when creating an output model, the model will inherit these properties.

        :param dict enumeration: A label enumeration dictionary of string (label) to integer (value) pairs.

            For example:

            .. code-block:: javascript

               {
                    "background": 0,
                    "person": 1
               }
        """
        super(Project, self).set_model_label_enumeration(enumeration=enumeration)

    def get_last_iteration(self):
        # type: () -> int
        """
        Get the last reported iteration, which is the last iteration for which the Project reported a metric.

        .. note::
           The maximum reported iteration is not in the local cache. This method
           sends a request to the **ClearML Server** (backend).

        :return: The last reported iteration number.
        """
        self._reload_last_iteration()
        return max(self.data.last_iteration or 0, self.__reporter.max_iteration if self.__reporter else 0)

    def set_initial_iteration(self, offset=0):
        # type: (int) -> int
        """
        Set initial iteration, instead of zero. Useful when continuing training from previous checkpoints

        :param int offset: Initial iteration (at starting point)
        :return: Newly set initial offset.
        """
        return super(Project, self).set_initial_iteration(offset=offset)

    def get_initial_iteration(self):
        # type: () -> int
        """
        Return the initial iteration offset, default is 0
        Useful when continuing training from previous checkpoints

        :return: Initial iteration offset.
        """
        return super(Project, self).get_initial_iteration()

    def get_last_scalar_metrics(self):
        # type: () -> Dict[str, Dict[str, Dict[str, float]]]
        """
        Get the last scalar metrics which the Project reported. This is a nested dictionary, ordered by title and series.

        For example:

        .. code-block:: javascript

           {
            "title": {
                "series": {
                    "last": 0.5,
                    "min": 0.1,
                    "max": 0.9
                    }
                }
            }

        :return: The last scalar metrics.
        """
        self.reload()
        metrics = self.data.last_metrics
        scalar_metrics = dict()
        for i in metrics.values():
            for j in i.values():
                scalar_metrics.setdefault(j['metric'], {}).setdefault(
                    j['variant'], {'last': j['value'], 'min': j['min_value'], 'max': j['max_value']})
        return scalar_metrics

    def get_parameters_as_dict(self, cast=False):
        # type: (bool) -> Dict
        """
        Get the Project parameters as a raw nested dictionary.

        .. note::
           If `cast` is False (default) The values are not parsed. They are returned as is.

        :param cast: If True, cast the parameter to the original type. Default False,
            values are returned in their string representation

        """
        return naive_nested_from_flat_dictionary(self.get_parameters(cast=cast))

    def set_parameters_as_dict(self, dictionary):
        # type: (Dict) -> None
        """
        Set the parameters for the Project object from a dictionary. The dictionary can be nested.
        This does not link the dictionary to the Project object. It does a one-time update. This
        is the same behavior as the :meth:`Project.connect` method.
        """
        self._arguments.copy_from_dict(flatten_dictionary(dictionary))

    def get_user_properties(self, value_only=False):
        # type: (bool) -> Dict[str, Union[str, dict]]
        """
        Get user properties for this project.
        Returns a dictionary mapping user property name to user property details dict.

        :param value_only: If True, returned user property details will be a string representing the property value.
        """
        if not Session.check_min_api_version("2.9"):
            self.log.info("User properties are not supported by the server")
            return {}

        section = "properties"

        params = self._hyper_params_manager.get_hyper_params(
            sections=[section], projector=(lambda x: x.get("value")) if value_only else None
        )

        return dict(params.get(section, {}))

    def set_user_properties(
            self,
            *iterables,  # type: Union[Mapping[str, Union[str, dict, None]], Iterable[dict]]
            **properties  # type: Union[str, dict, int, float, None]
    ):
        # type: (...) -> bool
        """
        Set user properties for this project.
        A user property can contain the following fields (all of type string):
        name / value / description / type

        Examples:

        .. code-block:: py

            project.set_user_properties(backbone='great', stable=True)
            project.set_user_properties(backbone={"type": int, "description": "network type", "value": "great"}, )
            project.set_user_properties(
                {"name": "backbone", "description": "network type", "value": "great"},
                {"name": "stable", "description": "is stable", "value": True},
            )

        :param iterables: Properties iterables, each can be:

            * A dictionary of string key (name) to either a string value (value) a dict (property details). If the value
                is a dict, it must contain a "value" field. For example:

                .. code-block:: javascript

                    {
                        "property_name": {"description": "This is a user property", "value": "property value"},
                        "another_property_name": {"description": "This is user property", "value": "another value"},
                        "yet_another_property_name": "some value"
                    }


            * An iterable of dicts (each representing property details). Each dict must contain a "name" field and a
                "value" field. For example:

                .. code-block:: javascript

                    [
                        {
                            "name": "property_name",
                            "description": "This is a user property",
                            "value": "property value"
                        },
                        {
                            "name": "another_property_name",
                            "description": "This is another user property",
                            "value": "another value"
                        }
                    ]

        :param properties: Additional properties keyword arguments. Key is the property name, and value can be
            a string (property value) or a dict (property details). If the value is a dict, it must contain a "value"
            field. For example:

            .. code-block:: javascript

                {
                    "property_name": "string as property value",
                    "another_property_name": {
                        "type": "string",
                        "description": "This is user property",
                        "value": "another value"
                    }
                }

        """
        if not Session.check_min_api_version("2.9"):
            self.log.info("User properties are not supported by the server")
            return False

        return self._hyper_params_manager.edit_hyper_params(
            iterables=list(properties.items()) + (
                list(iterables.items()) if isinstance(iterables, dict) else list(iterables)),
            replace='none',
            force_section="properties",
        )

    def get_script(self):
        # type: (...) -> Mapping[str, Optional[str]]
        """
        Get project's script details.

        Returns a dictionary containing the script details.

        :return: Dictionary with script properties e.g.

        .. code-block:: javascript

           {
                'working_dir': 'examples/reporting',
                'entry_point': 'artifacts.py',
                'branch': 'master',
                'repository': 'https://github.com/allegroai/clearml.git'
           }

        """
        script = self.data.script
        return {
            "working_dir": script.working_dir,
            "entry_point": script.entry_point,
            "branch": script.branch,
            "repository": script.repository
        }

    def set_script(
            self,
            repository=None,  # type: Optional[str]
            branch=None,  # type: Optional[str]
            commit=None,  # type: Optional[str]
            diff=None,  # type: Optional[str]
            working_dir=None,  # type: Optional[str]
            entry_point=None,  # type: Optional[str]
    ):
        # type: (...) -> None
        """
        Set project's script.

        Examples:

        .. code-block:: py

            project.set_script(
                repository='https://github.com/allegroai/clearml.git,
                branch='main',
                working_dir='examples/reporting',
                entry_point='artifacts.py'
            )

        :param repository: Optional, URL of remote repository. use empty string ("") to clear repository entry.
        :param branch: Optional, Select specific repository branch / tag. use empty string ("") to clear branch entry.
        :param commit: Optional, set specific git commit id. use empty string ("") to clear commit ID entry.
        :param diff: Optional, set "git diff" section. use empty string ("") to clear git-diff entry.
        :param working_dir: Optional, Working directory to launch the script from.
        :param entry_point: Optional, Path to execute within the repository.

        """
        self.reload()
        script = self.data.script
        if repository is not None:
            script.repository = str(repository)
        if branch is not None:
            script.branch = str(branch)
            if script.tag:
                script.tag = None
        if commit is not None:
            script.version_num = str(commit)
        if diff is not None:
            script.diff = str(diff)
        if working_dir is not None:
            script.working_dir = str(working_dir)
        if entry_point is not None:
            script.entry_point = str(entry_point)
        # noinspection PyProtectedMember
        self._update_script(script=script)

    def delete_user_properties(self, *iterables):
        # type: (Iterable[Union[dict, Iterable[str, str]]]) -> bool
        """
        Delete hyperparameters for this project.

        :param iterables: Hyperparameter key iterables. Each an iterable whose possible values each represent
            a hyperparameter entry to delete, value formats are:

            * A dictionary containing a 'section' and 'name' fields
            * An iterable (e.g. tuple, list etc.) whose first two items denote 'section' and 'name'
        """
        if not Session.check_min_api_version("2.9"):
            self.log.info("User properties are not supported by the server")
            return False

        return self._hyper_params_manager.delete_hyper_params(*iterables)

    def set_base_docker(
            self,
            docker_cmd=None,  # type: Optional[str]
            docker_image=None,  # type: Optional[str]
            docker_arguments=None,  # type: Optional[Union[str, Sequence[str]]]
            docker_setup_bash_script=None  # type: Optional[Union[str, Sequence[str]]]
    ):
        # type: (...) -> ()
        """
        Set the base docker image for this experiment
        If provided, this value will be used by clearml-agent to execute this experiment
        inside the provided docker image.
        When running remotely the call is ignored

        :param docker_cmd: Deprecated! compound docker container image + arguments
            (example: 'nvidia/cuda:11.1 -e test=1') Deprecated, use specific arguments.
        :param docker_image: docker container image (example: 'nvidia/cuda:11.1')
        :param docker_arguments: docker execution parameters (example: '-e ENV=1')
        :param docker_setup_bash_script: bash script to run at the
            beginning of the docker before launching the Project itself. example: ['apt update', 'apt-get install -y gcc']
        """
        if not self.running_locally() and self.is_main_project():
            return

        super(Project, self).set_base_docker(
            docker_cmd=docker_cmd or docker_image,
            docker_arguments=docker_arguments,
            docker_setup_bash_script=docker_setup_bash_script
        )

    @classmethod
    def set_resource_monitor_iteration_timeout(
        cls,
        seconds_from_start=30.0,
        wait_for_first_iteration_to_start_sec=180.0,
        max_wait_for_first_iteration_to_start_sec=1800.0,
    ):
        # type: (float, float, float) -> bool
        """
        Set the ResourceMonitor maximum duration (in seconds) to wait until first scalar/plot is reported.
        If timeout is reached without any reporting, the ResourceMonitor will start reporting machine statistics based
        on seconds from Project start time (instead of based on iteration).
        Notice! Should be called before `Project.init`.

        :param seconds_from_start: Maximum number of seconds to wait for scalar/plot reporting before defaulting
            to machine statistics reporting based on seconds from experiment start time
        :param wait_for_first_iteration_to_start_sec: Set the initial time (seconds) to wait for iteration reporting
            to be used as x-axis for the resource monitoring, if timeout exceeds then reverts to `seconds_from_start`
        :param max_wait_for_first_iteration_to_start_sec: Set the maximum time (seconds) to allow the resource
            monitoring to revert back to iteration reporting x-axis after starting to report `seconds_from_start`

        :return: True if success
        """
        if ResourceMonitor._resource_monitor_instances:
            getLogger().warning(
                "Project.set_resource_monitor_iteration_timeout called after Project.init."
                " This might not work since the values might not be used in forked processes"
            )
        # noinspection PyProtectedMember
        for instance in ResourceMonitor._resource_monitor_instances:
            # noinspection PyProtectedMember
            instance._first_report_sec = seconds_from_start
            instance.wait_for_first_iteration = wait_for_first_iteration_to_start_sec
            instance.max_check_first_iteration = max_wait_for_first_iteration_to_start_sec

        # noinspection PyProtectedMember
        ResourceMonitor._first_report_sec_default = seconds_from_start
        # noinspection PyProtectedMember
        ResourceMonitor._wait_for_first_iteration_to_start_sec_default = wait_for_first_iteration_to_start_sec
        # noinspection PyProtectedMember
        ResourceMonitor._max_wait_for_first_iteration_to_start_sec_default = max_wait_for_first_iteration_to_start_sec
        return True

    def execute_remotely(self, queue_name=None, clone=False, exit_process=True):
        # type: (Optional[str], bool, bool) -> Optional[Project]
        """
        If project is running locally (i.e., not by ``clearml-agent``), then clone the Project and enqueue it for remote
        execution; or, stop the execution of the current Project, reset its state, and enqueue it. If ``exit==True``,
        *exit* this process.

        .. note::
            If the project is running remotely (i.e., ``clearml-agent`` is executing it), this call is a no-op
            (i.e., does nothing).

        :param queue_name: The queue name used for enqueueing the project. If ``None``, this call exits the process
            without enqueuing the project.
        :param clone: Clone the Project and execute the newly cloned Project
            The values are:

          - ``True`` - A cloned copy of the Project will be created, and enqueued, instead of this Project.
          - ``False`` - The Project will be enqueued.

        :param exit_process: The function call will leave the calling process at the end.

          - ``True`` - Exit the process (exit(0)). Note: if ``clone==False``, then ``exit_process`` must be ``True``.
          - ``False`` - Do not exit the process.

        :return Project: return the project object of the newly generated remotely executing project
        """
        # do nothing, we are running remotely
        if running_remotely() and self.is_main_project():
            return None

        if not self.is_main_project():
            LoggerRoot.get_base_logger().warning(
                "Calling project.execute_remotely is only supported on main Project (created with Project.init)\n"
                "Defaulting to self.enqueue(queue_name={})".format(queue_name)
            )
            if not queue_name:
                raise ValueError("queue_name must be provided")
            enqueue_project = Project.clone(source_project=self) if clone else self
            Project.enqueue(project=enqueue_project, queue_name=queue_name)
            return

        if not clone and not exit_process:
            raise ValueError(
                "clone==False and exit_process==False is not supported. "
                "Project enqueuing itself must exit the process afterwards.")

        # make sure we analyze the process
        if self.status in (Project.ProjectStatusEnum.in_progress,):
            if clone:
                # wait for repository detection (5 minutes should be reasonable time to detect all packages)
                self.flush(wait_for_uploads=True)
                if self._logger and not self.__is_subprocess():
                    self._wait_for_repo_detection(timeout=300.)
            else:
                # close ourselves (it will make sure the repo is updated)
                self.close()

        # clone / reset Project
        if clone:
            project = Project.clone(self)
        else:
            project = self
            # check if the server supports enqueueing aborted/stopped Projects
            if Session.check_min_api_server_version('2.13'):
                self.mark_stopped(force=True)
            else:
                self.reset()

        # enqueue ourselves
        if queue_name:
            Project.enqueue(project, queue_name=queue_name)
            LoggerRoot.get_base_logger().warning(
                'Switching to remote execution, output log page {}'.format(project.get_output_log_web_page()))
        else:
            # Remove the development system tag
            system_tags = [t for t in project.get_system_tags() if t != self._development_tag]
            self.set_system_tags(system_tags)
            # if we leave the Project out there, it makes sense to make it editable.
            self.reset(force=True)

        # leave this process.
        if exit_process:
            LoggerRoot.get_base_logger().warning(
                'ClearML Terminating local execution process - continuing execution remotely')
            leave_process(0)

        return project

    def create_function_project(self, func, func_name=None, project_name=None, **kwargs):
        # type: (Callable, Optional[str], Optional[str], **Optional[Any]) -> Optional[Project]
        """
        Create a new project, and call ``func`` with the specified kwargs.
        One can think of this call as remote forking, where the newly created instance is the new Project
        calling the specified func with the appropriate kwargs and leaving once the func terminates.
        Notice that a remote executed function cannot create another child remote executed function.

        .. note::
            - Must be called from the main Project, i.e. the one created by Project.init(...)
            - The remote Projects inherits the environment from the creating Project
            - In the remote Project, the entrypoint is the same as the creating Project
            - In the remote Project, the execution is the same until reaching this function call

        :param func: A function to execute remotely as a single Project.
            On the remote executed Project the entry-point and the environment are copied from this
            calling process, only this function call redirect the execution flow to the called func,
            alongside the passed arguments
        :param func_name: A unique identifier of the function. Default the function name without the namespace.
            For example Class.foo() becomes 'foo'
        :param project_name: The newly created Project name. Default: the calling Project name + function name
        :param kwargs: name specific arguments for the target function.
            These arguments will appear under the configuration, "Function" section

        :return Project: Return the newly created Project or None if running remotely and execution is skipped
        """
        if not self.is_main_project():
            raise ValueError("Only the main Project object can call create_function_project()")
        if not callable(func):
            raise ValueError("func must be callable")
        if not Session.check_min_api_version('2.9'):
            raise ValueError("Remote function execution is not supported, "
                             "please upgrade to the latest server version")

        func_name = str(func_name or func.__name__).strip()
        if func_name in self._remote_functions_generated:
            raise ValueError("Function name must be unique, a function by the name '{}' "
                             "was already created by this Project.".format(func_name))

        section_name = 'Function'
        tag_name = 'func'
        func_marker = '__func_readonly__'

        # sanitize the dict, leave only basic types that we might want to override later in the UI
        func_params = {k: v for k, v in kwargs.items() if verify_basic_value(v)}
        func_params[func_marker] = func_name

        # do not query if we are running locally, there is no need.
        project_func_marker = self.running_locally() or self.get_parameter('{}/{}'.format(section_name, func_marker))

        # if we are running locally or if we are running remotely but we are not a forked projects
        # condition explained:
        # (1) running in development mode creates all the forked projects
        # (2) running remotely but this is not one of the forked projects (i.e. it is missing the fork tag attribute)
        if self.running_locally() or not project_func_marker:
            self._wait_for_repo_detection(300)
            project = self.clone(self, name=project_name or '{} <{}>'.format(self.name, func_name), parent=self.id)
            project.set_system_tags((project.get_system_tags() or []) + [tag_name])
            project.connect(func_params, name=section_name)
            self._remote_functions_generated[func_name] = project.id
            return project

        # check if we are one of the generated functions and if this is us,
        # if we are not the correct function, not do nothing and leave
        if project_func_marker != func_name:
            self._remote_functions_generated[func_name] = len(self._remote_functions_generated) + 1
            return

        # mark this is us:
        self._remote_functions_generated[func_name] = self.id

        # this is us for sure, let's update the arguments and call the function
        self.connect(func_params, name=section_name)
        func_params.pop(func_marker, None)
        kwargs.update(func_params)
        func(**kwargs)
        # This is it, leave the process
        leave_process(0)

    def wait_for_status(
            self,
            status=(_Project.ProjectStatusEnum.completed, _Project.ProjectStatusEnum.stopped, _Project.ProjectStatusEnum.closed),
            raise_on_status=(_Project.ProjectStatusEnum.failed,),
            check_interval_sec=60.,
    ):
        # type: (Iterable[Project.ProjectStatusEnum], Optional[Iterable[Project.ProjectStatusEnum]], float) -> ()
        """
        Wait for a project to reach a defined status.

        :param status: Status to wait for. Defaults to ('completed', 'stopped', 'closed', )
        :param raise_on_status: Raise RuntimeError if the status of the projects matches one of these values.
            Defaults to ('failed').
        :param check_interval_sec: Interval in seconds between two checks. Defaults to 60 seconds.

        :raise: RuntimeError if the status is one of ``{raise_on_status}``.
        """
        stopped_status = list(status) + (list(raise_on_status) if raise_on_status else [])
        while self.status not in stopped_status:
            time.sleep(check_interval_sec)

        if raise_on_status and self.status in raise_on_status:
            raise RuntimeError("Project {} has status: {}.".format(self.project_id, self.status))

        # make sure we have the Project object
        self.reload()

    def export_project(self):
        # type: () -> dict
        """
        Export Project's configuration into a dictionary (for serialization purposes).
        A Project can be copied/modified by calling Project.import_project()
        Notice: Export project does not include the projects outputs, such as results
        (scalar/plots etc.) or Project artifacts/models

        :return: dictionary of the Project's configuration.
        """
        self.reload()
        export_data = self.data.to_dict()
        export_data.pop('last_metrics', None)
        export_data.pop('last_iteration', None)
        export_data.pop('status_changed', None)
        export_data.pop('status_reason', None)
        export_data.pop('status_message', None)
        export_data.get('execution', {}).pop('artifacts', None)
        export_data.get('execution', {}).pop('model', None)
        export_data['project_name'] = self.get_project_name()
        export_data['session_api_version'] = self.session.api_version
        return export_data

    def update_project(self, project_data):
        # type: (dict) -> bool
        """
        Update current project with configuration found on the project_data dictionary.
        See also export_project() for retrieving Project configuration.

        :param project_data: dictionary with full Project configuration
        :return: return True if Project update was successful
        """
        return bool(self.import_project(project_data=project_data, target_project=self, update=True))

    def rename(self, new_name):
        # type: (str) -> bool
        """
        Rename this project

        :param new_name: The new name of this project

        :return: True if the rename was successful and False otherwise
        """
        result = bool(self._edit(name=new_name))
        self.reload()
        return result

    def move_to_project(self, new_project_id=None, new_project_name=None, system_tags=None):
        # type: (Optional[str], Optional[str], Optional[Sequence[str]]) -> bool
        """
        Move this project to another project

        :param new_project_id: The ID of the project the project should be moved to.
            Not required if `new_project_name` is passed.
        :param new_project_name: Name of the new project the project should be moved to.
            Not required if `new_project_id` is passed.
        :param system_tags: System tags for the project the project should be moved to.

        :return: True if the move was successful and False otherwise
        """
        new_project_id = get_or_create_project(
            self.session, project_name=new_project_name, project_id=new_project_id, system_tags=system_tags
        )
        result = bool(self._edit(project=new_project_id))
        self.reload()
        return result

    def register_abort_callback(
            self,
            callback_function,  # type: Optional[Callable]
            callback_execution_timeout=30.  # type: float
    ):  # type (...) -> None
        """
        Register a Project abort callback (single callback function support only).
        Pass a function to be called from a background thread when the Project is **externally** being aborted.
        Users must specify a timeout for the callback function execution (default 30 seconds)
        if the callback execution function exceeds the timeout, the Project's process will be terminated

        Call this register function from the main process only.

        Note: Ctrl-C is Not considered external, only backend induced abort is covered here

        :param callback_function: Callback function to be called via external thread (from the main process).
            pass None to remove existing callback
        :param callback_execution_timeout: Maximum callback execution time in seconds, after which the process
            will be terminated even if the callback did not return
        """
        if self.__is_subprocess():
            raise ValueError("Register abort callback must be called from the main process, this is a subprocess.")

        if callback_function is None:
            if self._dev_worker:
                self._dev_worker.register_abort_callback(callback_function=None, execution_timeout=0, poll_freq=0)
            return

        if float(callback_execution_timeout) <= 0:
            raise ValueError(
                "function_timeout_sec must be positive timeout in seconds, got {}".format(callback_execution_timeout))

        # if we are running remotely we might not have a DevWorker monitoring us, so let's create one
        if not self._dev_worker:
            self._dev_worker = DevWorker()
            self._dev_worker.register(self, stop_signal_support=True)

        poll_freq = 15.0
        self._dev_worker.register_abort_callback(
            callback_function=callback_function,
            execution_timeout=callback_execution_timeout,
            poll_freq=poll_freq
        )

    @classmethod
    def import_project(cls, project_data, target_project=None, update=False):
        # type: (dict, Optional[Union[str, Project]], bool) -> Optional[Project]
        """
        Import (create) Project from previously exported Project configuration (see Project.export_project)
        Can also be used to edit/update an existing Project (by passing `target_project` and `update=True`).

        :param project_data: dictionary of a Project's configuration
        :param target_project: Import project_data into an existing Project. Can be either project_id (str) or Project object.
        :param update: If True, merge project_data with current Project configuration.
        :return: return True if Project was imported/updated
        """

        # restore original API version (otherwise, we might not be able to restore the data correctly)
        force_api_version = project_data.get('session_api_version') or None
        original_api_version = Session.api_version
        original_force_max_api_version = Session.force_max_api_version
        if force_api_version:
            Session.force_max_api_version = str(force_api_version)

        if not target_project:
            project_name = project_data.get('project_name') or Project._get_project_name(project_data.get('project', ''))
            target_project = Project.create(project_name=project_name, project_name=project_data.get('name', None))
        elif isinstance(target_project, six.string_types):
            target_project = Project.get_project(project_id=target_project)  # type: Optional[Project]
        elif not isinstance(target_project, Project):
            raise ValueError(
                "`target_project` must be either Project id (str) or Project object, "
                "received `target_project` type {}".format(type(target_project)))
        target_project.reload()
        cur_data = target_project.data.to_dict()
        cur_data = merge_dicts(cur_data, project_data) if update else dict(**project_data)
        cur_data.pop('id', None)
        cur_data.pop('project', None)
        # noinspection PyProtectedMember
        valid_fields = list(projects.EditRequest._get_data_props().keys())
        cur_data = dict((k, cur_data[k]) for k in valid_fields if k in cur_data)
        res = target_project._edit(**cur_data)
        if res and res.ok():
            target_project.reload()
        else:
            target_project = None

        # restore current api version, and return a new instance if Project with the current version
        if force_api_version:
            Session.force_max_api_version = original_force_max_api_version
            Session.api_version = original_api_version
            if target_project:
                target_project = Project.get_project(project_id=target_project.id)

        return target_project

    @classmethod
    def set_offline(cls, offline_mode=False):
        # type: (bool) -> None
        """
        Set offline mode, where all data and logs are stored into local folder, for later transmission

        .. note::
            `Project.set_offline` can't move the same project from offline to online, nor can it be applied before `Project.create`.
            See below an example of **incorrect** usage of `Project.set_offline`:

            ```
            from clearml import Project

            Project.set_offline(True)
            project = Project.create(project_name='DEBUG', project_name="offline")
            # ^^^ an error or warning is raised, saying that Project.set_offline(True)
            #     is supported only for `Project.init`
            Project.set_offline(False)
            # ^^^ an error or warning is raised, saying that running Project.set_offline(False)
            #     while the current project is not closed is not supported

            data = project.export_project()

            imported_project = Project.import_project(project_data=data)
            ```

            The correct way to use `Project.set_offline` can be seen in the following example:

            ```
            from clearml import Project

            Project.set_offline(True)
            project = Project.init(project_name='DEBUG', project_name="offline")
            project.upload_artifact("large_artifact", "test_string")
            project.close()
            Project.set_offline(False)

            imported_project = Project.import_offline_session(project.get_offline_mode_folder())
            ```

        :param offline_mode: If True, offline-mode is turned on, and no communication to the backend is enabled.
        :return:
        """
        if running_remotely() or bool(offline_mode) == InterfaceBase._offline_mode:
            return
        if (
            cls.current_project()
            and cls.current_project().status != cls.ProjectStatusEnum.closed
            and not offline_mode
        ):
            raise UsageError(
                "Switching from offline mode to online mode, but the current project has not been closed. Use `Project.close` to close it."
            )
        ENV_OFFLINE_MODE.set(offline_mode)
        InterfaceBase._offline_mode = bool(offline_mode)
        Session._offline_mode = bool(offline_mode)
        if not offline_mode:
            # noinspection PyProtectedMember
            Session._make_all_sessions_go_online()

    @classmethod
    def is_offline(cls):
        # type: () -> bool
        """
        Return offline-mode state, If in offline-mode, no communication to the backend is enabled.

        :return: boolean offline-mode state
        """
        return cls._offline_mode

    @classmethod
    def import_offline_session(cls, session_folder_zip, previous_project_id=None, iteration_offset=0):
        # type: (str, Optional[str], Optional[int]) -> (Optional[str])
        """
        Upload an offline session (execution) of a Project.
        Full Project execution includes repository details, installed packages, artifacts, logs, metric and debug samples.
        This function may also be used to continue a previously executed project with a project executed offline.

        :param session_folder_zip: Path to a folder containing the session, or zip-file of the session folder.
        :param previous_project_id: Project ID of the project you wish to continue with this offline session.
        :param iteration_offset: Reporting of the offline session will be offset with the
            number specified by this parameter. Useful for avoiding overwriting metrics.

        :return: Newly created project ID or the ID of the continued project (previous_project_id)
        """
        print('ClearML: Importing offline session from {}'.format(session_folder_zip))

        temp_folder = None
        if Path(session_folder_zip).is_file():
            # unzip the file:
            temp_folder = mkdtemp(prefix='clearml-offline-')
            ZipFile(session_folder_zip).extractall(path=temp_folder)
            session_folder_zip = temp_folder

        session_folder = Path(session_folder_zip)
        if not session_folder.is_dir():
            raise ValueError("Could not find the session folder / zip-file {}".format(session_folder))

        try:
            with open((session_folder / cls._offline_filename).as_posix(), 'rt') as f:
                export_data = json.load(f)
        except Exception as ex:
            raise ValueError(
                "Could not read Project object {}: Exception {}".format(session_folder / cls._offline_filename, ex))
        current_project = cls.import_project(export_data)
        if previous_project_id:
            project_holding_reports = cls.get_project(project_id=previous_project_id)
            project_holding_reports.mark_started(force=True)
            project_holding_reports = cls.import_project(export_data, target_project=project_holding_reports, update=True)
        else:
            project_holding_reports = current_project
            project_holding_reports.mark_started(force=True)
        # fix artifacts
        if current_project.data.execution.artifacts:
            from . import StorageManager
            # noinspection PyProtectedMember
            offline_folder = os.path.join(export_data.get('offline_folder', ''), 'data/')

            # noinspection PyProtectedMember
            remote_url = current_project._get_default_report_storage_uri()
            if remote_url and remote_url.endswith('/'):
                remote_url = remote_url[:-1]

            for artifact in current_project.data.execution.artifacts:
                local_path = artifact.uri.replace(offline_folder, '', 1)
                local_file = session_folder / 'data' / local_path
                if local_file.is_file():
                    remote_path = local_path.replace(
                        '.{}{}'.format(export_data['id'], os.sep), '.{}{}'.format(current_project.id, os.sep), 1)
                    artifact.uri = '{}/{}'.format(remote_url, remote_path)
                    StorageManager.upload_file(local_file=local_file.as_posix(), remote_url=artifact.uri)
            # noinspection PyProtectedMember
            project_holding_reports._edit(execution=current_project.data.execution)
        for output_model in export_data.get("offline_output_models", []):
            model = OutputModel(project=current_project, **output_model["init"])
            if output_model.get("output_uri"):
                model.set_upload_destination(output_model.get("output_uri"))
            model.update_weights(auto_delete_file=False, **output_model["weights"])
            Metrics.report_offline_session(
                model,
                session_folder,
                iteration_offset=iteration_offset,
                remote_url=project_holding_reports._get_default_report_storage_uri(),
                only_with_id=output_model["id"],
                session=project_holding_reports.session
            )
        # logs
        ProjectHandler.report_offline_session(project_holding_reports, session_folder, iteration_offset=iteration_offset)
        # metrics
        Metrics.report_offline_session(
            project_holding_reports,
            session_folder,
            iteration_offset=iteration_offset,
            only_with_id=export_data["id"],
            session=project_holding_reports.session,
        )
        # print imported results page
        print('ClearML results page: {}'.format(project_holding_reports.get_output_log_web_page()))
        project_holding_reports.mark_completed()
        # close project
        project_holding_reports.close()

        # cleanup
        if temp_folder:
            # noinspection PyBroadException
            try:
                shutil.rmtree(temp_folder)
            except Exception:
                pass

        return project_holding_reports.id

    @classmethod
    def set_credentials(
            cls,
            api_host=None,
            web_host=None,
            files_host=None,
            key=None,
            secret=None,
            store_conf_file=False
    ):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], bool) -> None
        """
        Set new default **ClearML Server** (backend) host and credentials.

        These credentials will be overridden by either OS environment variables, or the ClearML configuration
        file, ``clearml.conf``.

        .. warning::
           Credentials must be set before initializing a Project object.

        For example, to set credentials for a remote computer:

        .. code-block:: py

            Project.set_credentials(
                api_host='http://localhost:8008', web_host='http://localhost:8080', files_host='http://localhost:8081',
                key='optional_credentials',  secret='optional_credentials'
            )
            project = Project.init('project name', 'experiment name')

        :param str api_host: The API server url. For example, ``host='http://localhost:8008'``
        :param str web_host: The Web server url. For example, ``host='http://localhost:8080'``
        :param str files_host: The file server url. For example, ``host='http://localhost:8081'``
        :param str key: The user key (in the key/secret pair). For example, ``key='thisisakey123'``
        :param str secret: The user secret (in the key/secret pair). For example, ``secret='thisisseceret123'``
        :param bool store_conf_file: If True, store the current configuration into the ~/clearml.conf file.
            If the configuration file exists, no change will be made (outputs a warning).
            Not applicable when running remotely (i.e. clearml-agent).
        """
        if api_host:
            Session.default_host = api_host
            if not running_remotely() and not ENV_HOST.get():
                ENV_HOST.set(api_host)
        if web_host:
            Session.default_web = web_host
            if not running_remotely() and not ENV_WEB_HOST.get():
                ENV_WEB_HOST.set(web_host)
        if files_host:
            Session.default_files = files_host
            if not running_remotely() and not ENV_FILES_HOST.get():
                ENV_FILES_HOST.set(files_host)
        if key:
            Session.default_key = key
            if not running_remotely():
                ENV_ACCESS_KEY.set(key)
        if secret:
            Session.default_secret = secret
            if not running_remotely():
                ENV_SECRET_KEY.set(secret)

        if store_conf_file and not running_remotely():
            active_conf_file = get_active_config_file()
            if active_conf_file:
                getLogger().warning(
                    'Could not store credentials in configuration file, '
                    '\'{}\' already exists'.format(active_conf_file))
            else:
                conf = {'api': dict(
                    api_server=Session.default_host,
                    web_server=Session.default_web,
                    files_server=Session.default_files,
                    credentials=dict(access_key=Session.default_key, secret_key=Session.default_secret))}
                with open(get_config_file(), 'wt') as f:
                    lines = json.dumps(conf, indent=4).split('\n')
                    f.write('\n'.join(lines[1:-1]))

    @classmethod
    def debug_simulate_remote_project(cls, project_id, reset_project=False):
        # type: (str, bool) -> ()
        """
        Simulate remote execution of a specified Project.
        This call will simulate the behaviour of your Project as if executed by the ClearML-Agent
        This means configurations will be coming from the backend server into the code
        (the opposite from manual execution, where the backend logs the code arguments)
        Use with care.

        :param project_id: Project ID to simulate, notice that all configuration will be taken from the specified
            Project, regardless of the code initial values, just like it as if executed by ClearML agent
        :param reset_project: If True, target Project, is automatically cleared / reset.
        """

        # if we are already running remotely, do nothing
        if running_remotely():
            return

        # verify Project ID exists
        project = Project.get_project(project_id=project_id)
        if not project:
            raise ValueError("Project ID '{}' could not be found".format(project_id))

        if reset_project:
            project.reset(set_started_on_success=False, force=True)

        from .config.remote import override_current_project_id
        from .config.defs import LOG_TO_BACKEND_ENV_VAR
        override_current_project_id(project_id)
        LOG_TO_BACKEND_ENV_VAR.set(True)
        DEBUG_SIMULATE_REMOTE_TASK.set(True)

    def get_executed_queue(self, return_name=False):
        # type: (bool) -> Optional[str]
        """
        Get the queue the project was executed on.

        :param return_name: If True, return the name of the queue. Otherwise, return its ID

        :return: Return the ID or name of the queue the project was executed on.
            If no queue was found, return None
        """
        queue_id = self.data.execution.queue
        if not return_name or not queue_id:
            return queue_id
        try:
            queue_name_result = Project._send(
                Project._get_default_session(),
                queues.GetByIdRequest(queue_id)
            )
            return queue_name_result.response.queue.name
        except Exception as e:
            getLogger().warning("Could not get name of queue with ID '{}': {}".format(queue_id, e))
            return None

    @classmethod
    def _create(cls, project_name=None, project_name=None, project_type=ProjectTypes.training):
        # type: (Optional[str], Optional[str], Project.ProjectTypes) -> ProjectInstance
        """
        Create a new unpopulated Project (experiment).

        :param str project_name: The name of the project in which the experiment will be created.
            If ``project_name`` is ``None``, and the main execution Project is initialized (see :meth:`Project.init`),
            then the main execution Project's project is used. Otherwise, if the project does
            not exist, it is created. (Optional)
        :param str project_name: The name of Project (experiment).
        :param ProjectTypes project_type: The project type.

        :return: The newly created project created.
        :rtype: Project
        """
        if not project_name:
            if not cls.__main_project:
                raise ValueError("Please provide project_name, no global project context found "
                                 "(Project.current_project hasn't been called)")
            project_name = cls.__main_project.get_project_name()

        try:
            project = cls(
                private=cls.__create_protection,
                project_name=project_name,
                project_name=project_name,
                project_type=project_type,
                log_to_backend=False,
                force_create=True,
            )
        except Exception:
            raise
        return project

    def _set_model_config(self, config_text=None, config_dict=None):
        # type: (Optional[str], Optional[Mapping]) -> None
        """
        Set Project model configuration text/dict

        :param config_text: model configuration (unconstrained text string). usually the content
            of a configuration file. If `config_text` is not None, `config_dict` must not be provided.
        :param config_dict: model configuration parameters dictionary.
            If `config_dict` is not None, `config_text` must not be provided.
        """
        # noinspection PyProtectedMember
        design = OutputModel._resolve_config(config_text=config_text, config_dict=config_dict)
        super(Project, self)._set_model_design(design=design)

    def _get_model_config_text(self):
        # type: () -> str
        """
        Get Project model configuration text (before creating an output model)
        When an output model is created it will inherit these properties

        :return: The model config_text (unconstrained text string).
        """
        return super(Project, self).get_model_design()

    def _get_model_config_dict(self):
        # type: () -> Dict
        """
        Get Project model configuration dictionary (before creating an output model)
        When an output model is created it will inherit these properties

        :return: config_dict: model configuration parameters dictionary.
        """
        config_text = self._get_model_config_text()
        # noinspection PyProtectedMember
        return OutputModel._text_to_config_dict(config_text)

    def _set_startup_info(self):
        # type: () -> ()
        self._set_runtime_properties(
            runtime_properties={"CLEARML VERSION": self.session.client, "CLI": sys.argv[0], "progress": "0"}
        )

    @classmethod
    def _reset_current_project_obj(cls):
        if not cls.__main_project:
            return
        project = cls.__main_project
        cls.__main_project = None
        cls.__forked_proc_main_pid = None
        if project._dev_worker:
            project._dev_worker.unregister()
            project._dev_worker = None

    @classmethod
    def _has_current_project_obj(cls):
        # type: () -> bool
        return bool(cls.__main_project)

    @classmethod
    def _create_dev_project(
            cls, default_project_name, default_project_name, default_project_type, tags,
            reuse_last_project_id, continue_last_project=False, detect_repo=True, auto_connect_streams=True
    ):
        if not default_project_name or not default_project_name:
            # get project name and project name from repository name and entry_point
            result, _ = ScriptInfo.get(create_requirements=False, check_uncommitted=False)
            if not default_project_name:
                # noinspection PyBroadException
                try:
                    parts = result.script['repository'].split('/')
                    default_project_name = (parts[-1] or parts[-2]).replace('.git', '') or 'Untitled'
                except Exception:
                    default_project_name = 'Untitled'
            if not default_project_name:
                # noinspection PyBroadException
                try:
                    default_project_name = os.path.splitext(os.path.basename(result.script['entry_point']))[0]
                except Exception:
                    pass

        # conform reuse_last_project_id and continue_last_project
        if continue_last_project and isinstance(continue_last_project, str):
            reuse_last_project_id = continue_last_project
            continue_last_project = True
        elif isinstance(continue_last_project, int) and continue_last_project is not True:
            # allow initial offset environment override
            continue_last_project = continue_last_project

        if TASK_SET_ITERATION_OFFSET.get() is not None:
            continue_last_project = TASK_SET_ITERATION_OFFSET.get()

        # if we force no project reuse from os environment
        if DEV_TASK_NO_REUSE.get() or not reuse_last_project_id or isinstance(reuse_last_project_id, str):
            default_project = None
        else:
            # if we have a previous session to use, get the project id from it
            default_project = cls.__get_last_used_project_id(
                default_project_name,
                default_project_name,
                default_project_type.value,
            )

        closed_old_project = False
        default_project_id = None
        project = None
        in_dev_mode = not running_remotely()

        if in_dev_mode:
            if isinstance(reuse_last_project_id, str) and reuse_last_project_id:
                default_project_id = reuse_last_project_id
            elif not reuse_last_project_id or not cls.__project_is_relevant(default_project):
                default_project_id = None
            else:
                default_project_id = default_project.get('id') if default_project else None

            if default_project_id:
                try:
                    project = cls(
                        private=cls.__create_protection,
                        project_id=default_project_id,
                        log_to_backend=True,
                    )

                    # instead of resting the previously used project we are continuing the training with it.
                    if project and \
                            (continue_last_project or
                             (isinstance(continue_last_project, int) and not isinstance(continue_last_project, bool))):
                        project.reload()
                        project.mark_started(force=True)
                        # allow to disable the
                        if continue_last_project is True:
                            project.set_initial_iteration(project.get_last_iteration() + 1)
                        else:
                            project.set_initial_iteration(continue_last_project)

                    else:
                        project_tags = project.data.system_tags if hasattr(project.data, 'system_tags') else project.data.tags
                        project_artifacts = project.data.execution.artifacts \
                            if hasattr(project.data.execution, 'artifacts') else None
                        if ((project._status in (
                                cls.ProjectStatusEnum.published, cls.ProjectStatusEnum.closed))
                                or project.output_models_id or (cls.archived_tag in project_tags)
                                or (cls._development_tag not in project_tags)
                                or project_artifacts):
                            # If the project is published or closed, we shouldn't reset it so we can't use it in dev mode
                            # If the project is archived, or already has an output model,
                            #  we shouldn't use it in development mode either
                            default_project_id = None
                            project = None
                        else:
                            with project._edit_lock:
                                # from now on, there is no need to reload, we just clear stuff,
                                # this flag will be cleared off once we actually refresh at the end of the function
                                project._reload_skip_flag = True
                                # reset the project, so we can update it
                                project.reset(set_started_on_success=False, force=False)
                                # clear the heaviest stuff first
                                project._clear_project(
                                    system_tags=[cls._development_tag],
                                    comment=make_message('Auto-generated at %(time)s by %(user)s@%(host)s'))

                except (Exception, ValueError):
                    # we failed reusing project, create a new one
                    default_project_id = None

        # create a new project
        if not default_project_id:
            project = cls(
                private=cls.__create_protection,
                project_name=default_project_name,
                project_name=default_project_name,
                project_type=default_project_type,
                log_to_backend=True,
            )
            # no need to reload yet, we clear this before the end of the function
            project._reload_skip_flag = True

        if in_dev_mode:
            # update this session, for later use
            cls.__update_last_used_project_id(default_project_name, default_project_name, default_project_type.value, project.id)
            # set default docker image from env.
            project._set_default_docker_image()

        # mark us as the main Project, there should only be one dev Project at a time.
        if not Project.__main_project:
            Project.__forked_proc_main_pid = os.getpid()
            Project.__main_project = project

        # mark the project as started
        project.started()
        # reload, making sure we are synced
        project._reload_skip_flag = False
        project.reload()

        # add Project tags
        if tags:
            project.add_tags([tags] if isinstance(tags, str) else tags)

        # force update of base logger to this current project (this is the main logger project)
        logger = project._get_logger(auto_connect_streams=auto_connect_streams)
        if closed_old_project:
            logger.report_text('ClearML Project: Closing old development project id={}'.format(default_project.get('id')))
        # print warning, reusing/creating a project
        if default_project_id and not continue_last_project:
            logger.report_text('ClearML Project: overwriting (reusing) project id=%s' % project.id)
        elif default_project_id and continue_last_project:
            logger.report_text('ClearML Project: continuing previous project id=%s '
                               'Notice this run will not be reproducible!' % project.id)
        else:
            logger.report_text('ClearML Project: created new project id=%s' % project.id)

        # update current repository and put warning into logs
        if detect_repo:
            # noinspection PyBroadException
            try:
                import traceback
                stack = traceback.extract_stack(limit=10)
                # NOTICE WE ARE ALWAYS 3 down from caller in stack!
                for i in range(len(stack) - 1, 0, -1):
                    # look for the Project.init call, then the one above it is the callee module
                    if stack[i].name == 'init':
                        project._calling_filename = os.path.abspath(stack[i - 1].filename)
                        break
            except Exception:
                pass
            if in_dev_mode and cls.__detect_repo_async:
                project._detect_repo_async_thread = threading.Thread(target=project._update_repository)
                project._detect_repo_async_thread.daemon = True
                project._detect_repo_async_thread.start()
            else:
                project._update_repository()

        # make sure we see something in the UI
        thread = threading.Thread(target=LoggerRoot.flush)
        thread.daemon = True
        thread.start()

        return project

    def _get_logger(self, flush_period=NotSet, auto_connect_streams=False):
        # type: (Optional[float], Union[bool, dict]) -> Logger
        """
        get a logger object for reporting based on the project

        :param flush_period: The period of the logger flush.
            If None of any other False value, will not flush periodically.
            If a logger was created before, this will be the new period and
            the old one will be discarded.

        :return: Logger object
        """

        if not self._logger:
            # do not recreate logger after project was closed/quit
            if self._at_exit_called and self._at_exit_called in (True, get_current_thread_id(),):
                raise ValueError("Cannot use Project Logger after project was closed")
            # Get a logger object
            self._logger = Logger(
                private_project=self,
                connect_stdout=(auto_connect_streams is True) or
                               (isinstance(auto_connect_streams, dict) and auto_connect_streams.get('stdout', False)),
                connect_stderr=(auto_connect_streams is True) or
                               (isinstance(auto_connect_streams, dict) and auto_connect_streams.get('stderr', False)),
                connect_logging=isinstance(auto_connect_streams, dict) and auto_connect_streams.get('logging', False),
            )
            # make sure we set our reported to async mode
            # we make sure we flush it in self._at_exit
            self._reporter.async_enable = True
            # if we just created the logger, set default flush period
            if not flush_period or flush_period is self.NotSet:
                flush_period = float(DevWorker.report_period)

        if isinstance(flush_period, (int, float)):
            flush_period = int(abs(flush_period))

        if flush_period is None or isinstance(flush_period, int):
            self._logger.set_flush_period(flush_period)

        return self._logger

    def _connect_output_model(self, model, name=None, **kwargs):
        assert isinstance(model, OutputModel)
        model.connect(self, name=name, ignore_remote_overrides=False)
        return model

    def _save_output_model(self, model):
        """
        Deprecated: Save a reference to the connected output model.

        :param model: The connected output model
        """
        # deprecated
        self._connected_output_model = model

    def _handle_ignore_remote_overrides(self, overrides_name, ignore_remote_overrides):
        if self.running_locally() and ignore_remote_overrides:
            self.set_parameter(
                overrides_name,
                True,
                description="If True, ignore UI/backend overrides when running remotely."
                " Set it to False if you would like the overrides to be applied",
                value_type=bool
            )
        elif not self.running_locally():
            ignore_remote_overrides = self.get_parameter(overrides_name, default=ignore_remote_overrides, cast=True)
        return ignore_remote_overrides

    def _reconnect_output_model(self):
        """
        Deprecated: If there is a saved connected output model, connect it again.

        This is needed if the input model is connected after the output model
        is connected, an then we will have to get the model design from the
        input model by reconnecting.
        """
        # Deprecated:
        if self._connected_output_model:
            self.connect(self._connected_output_model)

    def _connect_input_model(self, model, name=None, ignore_remote_overrides=False):
        assert isinstance(model, InputModel)
        # we only allow for an input model to be connected once
        # at least until we support multiple input models
        # notice that we do not check the project's input model because we allow project reuse and overwrite
        # add into comment that we are using this model

        # refresh comment
        comment = self._reload_field("comment") or self.comment or ''

        if not comment.endswith('\n'):
            comment += '\n'
        comment += 'Using model id: {}'.format(model.id)
        self.set_comment(comment)

        model.connect(self, name, ignore_remote_overrides=ignore_remote_overrides)
        return model

    def _connect_argparse(
        self, parser, args=None, namespace=None, parsed_args=None, name=None, ignore_remote_overrides=False
    ):
        # do not allow argparser to connect to jupyter notebook
        # noinspection PyBroadException
        try:
            if "IPython" in sys.modules:
                # noinspection PyPackageRequirements
                from IPython import get_ipython  # noqa

                ip = get_ipython()
                if ip is not None and "IPKernelApp" in ip.config:
                    return parser
        except Exception:
            pass

        if self.is_main_project():
            argparser_update_currentproject(self)

        if (parser is None or parsed_args is None) and argparser_parseargs_called():
            # if we have a parser but nor parsed_args, we need to find the parser
            if parser and not parsed_args:
                for _parser, _parsed_args in get_argparser_last_args():
                    if _parser == parser:
                        parsed_args = _parsed_args
                        break
            else:
                # prefer the first argparser (hopefully it is more relevant?!
                for _parser, _parsed_args in get_argparser_last_args():
                    if parser is None:
                        parser = _parser
                    if parsed_args is None and parser == _parser:
                        parsed_args = _parsed_args

        if running_remotely() and (self.is_main_project() or self._is_remote_main_project()) and not ignore_remote_overrides:
            self._arguments.copy_to_parser(parser, parsed_args)
        else:
            self._arguments.copy_defaults_from_argparse(
                parser, args=args, namespace=namespace, parsed_args=parsed_args)
        return parser

    def _connect_dictionary(self, dictionary, name=None, ignore_remote_overrides=False):
        def _update_args_dict(project, config_dict):
            # noinspection PyProtectedMember
            project._arguments.copy_from_dict(flatten_dictionary(config_dict), prefix=name)

        def _refresh_args_dict(project, config_proxy_dict):
            # reread from project including newly added keys
            # noinspection PyProtectedMember
            a_flat_dict = project._arguments.copy_to_dict(flatten_dictionary(config_proxy_dict), prefix=name)
            # noinspection PyProtectedMember
            nested_dict = config_proxy_dict._to_dict()
            config_proxy_dict.clear()
            config_proxy_dict._do_update(nested_from_flat_dictionary(nested_dict, a_flat_dict))

        def _check_keys(dict_, warning_sent=False):
            if warning_sent:
                return
            for k, v in dict_.items():
                if warning_sent:
                    return
                if not isinstance(k, str):
                    getLogger().warning(
                        "Unsupported key of type '{}' found when connecting dictionary. It will be converted to str".format(
                            type(k)
                        )
                    )
                    warning_sent = True
                if isinstance(v, dict):
                    _check_keys(v, warning_sent)

        if not running_remotely() or not (self.is_main_project() or self._is_remote_main_project()) or ignore_remote_overrides:
            _check_keys(dictionary)
            flat_dict = {str(k): v for k, v in flatten_dictionary(dictionary).items()}
            self._arguments.copy_from_dict(flat_dict, prefix=name)
            dictionary = ProxyDictPostWrite(self, _update_args_dict, **dictionary)
        else:
            flat_dict = flatten_dictionary(dictionary)
            flat_dict = self._arguments.copy_to_dict(flat_dict, prefix=name)
            dictionary = nested_from_flat_dictionary(dictionary, flat_dict)
            dictionary = ProxyDictPostWrite(self, _refresh_args_dict, **dictionary)

        return dictionary

    def _connect_project_parameters(self, attr_class, name=None, ignore_remote_overrides=False):
        ignore_remote_overrides_section = "_ignore_remote_overrides_"
        if running_remotely():
            ignore_remote_overrides = self.get_parameter(
                (name or "General") + "/" + ignore_remote_overrides_section, default=ignore_remote_overrides, cast=True
            )
        if running_remotely() and (self.is_main_project() or self._is_remote_main_project()) and not ignore_remote_overrides:
            parameters = self.get_parameters(cast=True)
            if name:
                parameters = dict(
                    (k[len(name) + 1:], v) for k, v in parameters.items() if k.startswith("{}/".format(name))
                )
            parameters.pop(ignore_remote_overrides_section, None)
            attr_class.update_from_dict(parameters)
        else:
            parameters_dict = attr_class.to_dict()
            if ignore_remote_overrides:
                parameters_dict[ignore_remote_overrides_section] = True
            self.set_parameters(parameters_dict, __parameters_prefix=name)
        return attr_class

    def _connect_object(self, an_object, name=None, ignore_remote_overrides=False):
        def verify_type(key, value):
            if str(key).startswith('_') or not isinstance(value, self._parameters_allowed_types):
                return False
            # verify everything is json able (i.e. basic types)
            try:
                json.dumps(value)
                return True
            except TypeError:
                return False

        a_dict = {
            k: v
            for cls_ in getattr(an_object, "__mro__", [an_object])
            for k, v in cls_.__dict__.items()
            if verify_type(k, v)
        }
        if running_remotely() and (self.is_main_project() or self._is_remote_main_project()) and not ignore_remote_overrides:
            a_dict = self._connect_dictionary(a_dict, name, ignore_remote_overrides=ignore_remote_overrides)
            for k, v in a_dict.items():
                if getattr(an_object, k, None) != a_dict[k]:
                    setattr(an_object, k, v)

            return an_object
        else:
            self._connect_dictionary(a_dict, name, ignore_remote_overrides=ignore_remote_overrides)
            return an_object

    def _dev_mode_stop_project(self, stop_reason, pid=None):
        # make sure we do not get called (by a daemon thread) after at_exit
        if self._at_exit_called:
            return

        self.log.warning(
            "### TASK STOPPED - USER ABORTED - {} ###".format(
                stop_reason.upper().replace('_', ' ')
            )
        )
        self.flush(wait_for_uploads=True)

        # if running remotely, we want the daemon to kill us
        if self.running_locally():
            self.stopped(status_reason='USER ABORTED')

        if self._dev_worker:
            self._dev_worker.unregister()

        # NOTICE! This will end the entire execution tree!
        if self.__exit_hook:
            self.__exit_hook.remote_user_aborted = True
        self._kill_all_child_processes(send_kill=False, pid=pid, allow_kill_calling_pid=False)
        time.sleep(2.0)
        self._kill_all_child_processes(send_kill=True, pid=pid, allow_kill_calling_pid=True)
        os._exit(1)  # noqa

    @staticmethod
    def _kill_all_child_processes(send_kill=False, pid=None, allow_kill_calling_pid=True):
        # get current process if pid not provided
        current_pid = os.getpid()
        kill_ourselves = None
        pid = pid or current_pid
        try:
            parent = psutil.Process(pid)
        except psutil.Error:
            # could not find parent process id
            return
        for child in parent.children(recursive=True):
            # kill ourselves last (if we need to)
            if child.pid == current_pid:
                kill_ourselves = child
                continue
            if send_kill:
                child.kill()
            else:
                child.terminate()

        # parent ourselves
        if allow_kill_calling_pid or parent.pid != current_pid:
            if send_kill:
                parent.kill()
            else:
                parent.terminate()

        # kill ourselves if we need to:
        if allow_kill_calling_pid and kill_ourselves:
            if send_kill:
                kill_ourselves.kill()
            else:
                kill_ourselves.terminate()

    def _dev_mode_setup_worker(self):
        if (running_remotely() and not DEBUG_SIMULATE_REMOTE_TASK.get()) \
                or not self.is_main_project() or self._at_exit_called or self._offline_mode:
            return

        if self._dev_worker:
            return self._dev_worker

        self._dev_worker = DevWorker()
        self._dev_worker.register(self)

        logger = self.get_logger()
        flush_period = logger.get_flush_period()
        if not flush_period or flush_period > self._dev_worker.report_period:
            logger.set_flush_period(self._dev_worker.report_period)

    def _wait_for_repo_detection(self, timeout=None):
        # wait for detection repo sync
        if not self._detect_repo_async_thread:
            return
        with self._repo_detect_lock:
            if not self._detect_repo_async_thread:
                return
            # noinspection PyBroadException
            try:
                if self._detect_repo_async_thread.is_alive():
                    # if negative timeout, just kill the thread:
                    if timeout is not None and timeout < 0:
                        from .utilities.lowlevel.threads import kill_thread
                        kill_thread(self._detect_repo_async_thread)
                    else:
                        self.log.info('Waiting for repository detection and full package requirement analysis')
                        self._detect_repo_async_thread.join(timeout=timeout)
                        # because join has no return value
                        if self._detect_repo_async_thread.is_alive():
                            self.log.info('Repository and package analysis timed out ({} sec), '
                                          'giving up'.format(timeout))
                            # done waiting, kill the thread
                            from .utilities.lowlevel.threads import kill_thread
                            kill_thread(self._detect_repo_async_thread)
                        else:
                            self.log.info('Finished repository detection and package analysis')
                self._detect_repo_async_thread = None
            except Exception:
                pass

    def _summary_artifacts(self):
        # signal artifacts upload, and stop daemon
        self._artifacts_manager.stop(wait=True)
        # print artifacts summary (if not empty)
        if self._artifacts_manager.summary:
            self.get_logger().report_text(self._artifacts_manager.summary)

    def _at_exit(self):
        # protect sub-process at_exit (should never happen)
        if self._at_exit_called and self._at_exit_called != get_current_thread_id():
            return

        # make sure we do not try to use events, because Python might deadlock itself.
        # https://bugs.python.org/issue41606
        if self.__is_subprocess():
            BackgroundMonitor.set_at_exit_state(True)

        # shutdown will clear the main, so we have to store it before.
        # is_main = self.is_main_project()
        # fix debugger signal in the middle, catch everything
        try:
            self.__shutdown()
        except:  # noqa
            pass
        # In rare cases we might need to forcefully shutdown the process, currently we should avoid it.
        # if is_main:
        #     # we have to forcefully shutdown if we have forked processes, sometimes they will get stuck
        #     os._exit(self.__exit_hook.exit_code if self.__exit_hook and self.__exit_hook.exit_code else 0)

    def __shutdown(self):
        """
        Will happen automatically once we exit code, i.e. atexit
        :return:
        """
        # protect sub-process at_exit
        if self._at_exit_called:
            is_sub_process = self.__is_subprocess()
            # if we are called twice (signal in the middle of the shutdown),
            _nested_shutdown_call = bool(self._at_exit_called == get_current_thread_id())
            if _nested_shutdown_call and not is_sub_process:
                # if we were called again in the main thread on the main process, let's try again
                # make sure we only do this once
                self._at_exit_called = True
            else:
                # make sure we flush stdout, this is the best we can do.
                if _nested_shutdown_call and self._logger and is_sub_process:
                    # noinspection PyProtectedMember
                    self._logger._close_stdout_handler(wait=True)
                    self._at_exit_called = True
                # if we get here, we should do nothing and leave
                return
        else:
            # from here only a single thread can re-enter
            self._at_exit_called = get_current_thread_id()

        LoggerRoot.clear_logger_handlers()

        # disable lock on signal callbacks, to avoid deadlocks.
        if self.__exit_hook and self.__exit_hook.signal is not None:
            self.__edit_lock = False

        is_sub_process = self.__is_subprocess()

        project_status = None
        # noinspection PyBroadException
        try:
            wait_for_uploads = True
            # first thing mark project as stopped, so we will not end up with "running" on lost projects
            # if we are running remotely, the daemon will take care of it
            wait_for_std_log = True
            if (not running_remotely() or DEBUG_SIMULATE_REMOTE_TASK.get()) \
                    and self.is_main_project() and not is_sub_process:
                # check if we crashed, ot the signal is not interrupt (manual break)
                project_status = ('stopped',)
                if self.__exit_hook:
                    is_exception = self.__exit_hook.exception
                    # check if we are running inside a debugger
                    if not is_exception and sys.modules.get('pydevd'):
                        # noinspection PyBroadException
                        try:
                            is_exception = sys.last_type
                        except Exception:
                            pass

                        # check if this is Jupyter interactive session, do not mark as exception
                        if 'IPython' in sys.modules:
                            is_exception = None

                    # only if we have an exception (and not ctrl-break) or signal is not SIGTERM / SIGINT
                    if (is_exception and not isinstance(is_exception, KeyboardInterrupt)
                        and is_exception != KeyboardInterrupt) \
                            or (not self.__exit_hook.remote_user_aborted and
                                (self.__exit_hook.signal not in (None, 2, 15) or self.__exit_hook.exit_code)):
                        project_status = (
                            'failed',
                            'Exception {}'.format(is_exception) if is_exception else
                            'Signal {}'.format(self.__exit_hook.signal))
                        wait_for_uploads = False
                    else:
                        wait_for_uploads = (self.__exit_hook.remote_user_aborted or self.__exit_hook.signal is None)
                        if not self.__exit_hook.remote_user_aborted and self.__exit_hook.signal is None and \
                                not is_exception:
                            project_status = ('completed',)
                        else:
                            project_status = ('stopped',)
                            # user aborted. do not bother flushing the stdout logs
                            wait_for_std_log = self.__exit_hook.signal is not None

            # wait for repository detection (if we didn't crash)
            if wait_for_uploads and self._logger:
                # we should print summary here
                self._summary_artifacts()
                # make sure that if we crashed the thread we are not waiting forever
                if not is_sub_process:
                    self._wait_for_repo_detection(timeout=10.)

            # kill the repo thread (negative timeout, do not wait), if it hasn't finished yet.
            if not is_sub_process:
                self._wait_for_repo_detection(timeout=-1)

            # wait for uploads
            print_done_waiting = False
            if wait_for_uploads and (BackendModel.get_num_results() > 0 or
                                     (self.__reporter and self.__reporter.events_waiting())):
                self.log.info('Waiting to finish uploads')
                print_done_waiting = True
            # from here, do not send log in background thread
            if wait_for_uploads:
                self.flush(wait_for_uploads=True)
                # wait until the reporter flush everything
                if self.__reporter:
                    self.__reporter.stop()
                    if self.is_main_project():
                        # notice: this will close the reporting for all the Projects in the system
                        Metrics.close_async_threads()
                        # notice: this will close the jupyter monitoring
                        ScriptInfo.close()
                if self.is_main_project():
                    # noinspection PyBroadException
                    try:
                        from .storage.helper import StorageHelper
                        StorageHelper.close_async_threads()
                    except Exception:
                        pass

                if print_done_waiting:
                    self.log.info('Finished uploading')
            # elif self._logger:
            #     # noinspection PyProtectedMember
            #     self._logger._flush_stdout_handler()

            # from here, do not check worker status
            if self._dev_worker:
                self._dev_worker.unregister()
                self._dev_worker = None

            # stop resource monitoring
            if self._resource_monitor:
                self._resource_monitor.stop()
                self._resource_monitor = None

            if self._logger:
                self._logger.set_flush_period(None)
                # noinspection PyProtectedMember
                self._logger._close_stdout_handler(wait=wait_for_uploads or wait_for_std_log)

            if not is_sub_process:
                # change project status
                if not project_status:
                    pass
                elif project_status[0] == 'failed':
                    self.mark_failed(status_reason=project_status[1])
                elif project_status[0] == 'completed':
                    self.set_progress(100)
                    self.mark_completed()
                elif project_status[0] == 'stopped':
                    self.stopped()

            # this is so in theory we can close a main project and start a new one
            if self.is_main_project():
                Project.__main_project = None
                Project.__forked_proc_main_pid = None
                Project.__update_master_pid_project(project=None)
        except Exception:
            # make sure we do not interrupt the exit process
            pass

        # make sure we store last project state
        if self._offline_mode and not is_sub_process:
            # noinspection PyBroadException
            try:
                # make sure the state of the offline data is saved
                self._edit()
                # create zip file
                offline_folder = self.get_offline_mode_folder()
                zip_file = offline_folder.as_posix() + '.zip'
                with ZipFile(zip_file, 'w', allowZip64=True, compression=ZIP_DEFLATED) as zf:
                    for filename in offline_folder.rglob('*'):
                        if filename.is_file():
                            relative_file_name = filename.relative_to(offline_folder).as_posix()
                            zf.write(filename.as_posix(), arcname=relative_file_name)
                print('ClearML Project: Offline session stored in {}'.format(zip_file))
            except Exception:
                pass

        # delete locking object (lock file)
        if self._edit_lock:
            # noinspection PyBroadException
            try:
                del self.__edit_lock
            except Exception:
                pass
            self._edit_lock = None

        # make sure no one will re-enter the shutdown method
        self._at_exit_called = True
        if not is_sub_process and BackgroundMonitor.is_subprocess_enabled():
            BackgroundMonitor.wait_for_sub_process(self)

        # we are done
        return

    @classmethod
    def _remove_exception_hooks(cls):
        if cls.__exit_hook:
            cls.__exit_hook.remove_exception_hooks()

    @classmethod
    def _remove_signal_hooks(cls):
        if cls.__exit_hook:
            cls.__exit_hook.remove_signal_hooks()

    @classmethod
    def __register_at_exit(cls, exit_callback):
        if cls.__exit_hook is None:
            # noinspection PyBroadException
            try:
                cls.__exit_hook = ExitHooks(exit_callback)
                cls.__exit_hook.hook()
            except Exception:
                cls.__exit_hook = None
        else:
            cls.__exit_hook.update_callback(exit_callback)

    @classmethod
    def __get_project(
            cls,
            project_id=None,  # type: Optional[str]
            project_name=None,  # type: Optional[str]
            project_name=None,  # type: Optional[str]
            include_archived=True,  # type: bool
            tags=None,  # type: Optional[Sequence[str]]
            project_filter=None  # type: Optional[dict]
    ):
        # type: (...) -> ProjectInstance

        if project_id:
            return cls(private=cls.__create_protection, project_id=project_id, log_to_backend=False)

        if project_name:
            res = cls._send(
                cls._get_default_session(),
                projects.GetAllRequest(
                    name=exact_match_regex(project_name)
                )
            )
            project = get_single_result(entity='project', query=project_name, results=res.response.projects)
        else:
            project = None

        # get default session, before trying to access projects.Project so that we do not create two sessions.
        session = cls._get_default_session()
        system_tags = 'system_tags' if hasattr(projects.Project, 'system_tags') else 'tags'
        project_filter = project_filter or {}
        if not include_archived:
            project_filter['system_tags'] = (project_filter.get('system_tags') or []) + ['-{}'.format(cls.archived_tag)]
        if tags:
            project_filter['tags'] = (project_filter.get('tags') or []) + list(tags)
        res = cls._send(
            session,
            projects.GetAllRequest(
                project=[project.id] if project else None,
                name=exact_match_regex(project_name) if project_name else None,
                only_fields=['id', 'name', 'last_update', system_tags],
                **project_filter
            )
        )
        res_projects = res.response.projects
        # if we have more than one result, filter out the 'archived' results
        # notice that if we only have one result we do get the archived one as well.
        if len(res_projects) > 1:
            filtered_projects = [t for t in res_projects if not getattr(t, system_tags, None) or
                              cls.archived_tag not in getattr(t, system_tags, None)]
            # if we did not filter everything (otherwise we have only archived projects, so we return them)
            if filtered_projects:
                res_projects = filtered_projects

        project = get_single_result(
            entity='project',
            query={k: v for k, v in dict(
                project_name=project_name, project_name=project_name, tags=tags,
                include_archived=include_archived, project_filter=project_filter).items() if v},
            results=res_projects, raise_on_error=False)
        if not project:
            # should never happen
            return None  # noqa

        return cls(
            private=cls.__create_protection,
            project_id=project.id,
            log_to_backend=False,
        )

    @classmethod
    def __get_projects(
        cls,
        project_ids=None,  # type: Optional[Sequence[str]]
        project_name=None,  # type: Optional[Union[Sequence[str],str]]
        project_name=None,  # type: Optional[str]
        **kwargs  # type: Any
    ):
        # type: (...) -> List[Project]

        if project_ids:
            if isinstance(project_ids, six.string_types):
                project_ids = [project_ids]
            return [cls(private=cls.__create_protection, project_id=project_id, log_to_backend=False) for project_id in project_ids]

        queried_projects = cls._query_projects(
            project_name=project_name, project_name=project_name, fetch_only_first_page=True, **kwargs
        )
        if len(queried_projects) == 500:
            LoggerRoot.get_base_logger().warning(
                "Too many requests when calling Project.get_projects()."
                " Returning only the first 500 results."
                " Use Project.query_projects() to fetch all project IDs"
            )
        return [cls(private=cls.__create_protection, project_id=project.id, log_to_backend=False) for project in queried_projects]

    @classmethod
    def _query_projects(
        cls,
        project_ids=None,
        project_name=None,
        project_name=None,
        fetch_only_first_page=False,
        exact_match_regex_flag=True,
        **kwargs
    ):
        res = None
        if not project_ids:
            project_ids = None
        elif isinstance(project_ids, six.string_types):
            project_ids = [project_ids]

        if project_name and isinstance(project_name, str):
            project_names = [project_name]
        else:
            project_names = project_name

        project_ids = []
        projects_not_found = []
        if project_names:
            for name in project_names:
                aux_kwargs = {}
                if kwargs.get("_allow_extra_fields_"):
                    aux_kwargs["_allow_extra_fields_"] = True
                    aux_kwargs["search_hidden"] = kwargs.get("search_hidden", False)
                res = cls._send(
                    cls._get_default_session(),
                    projects.GetAllRequest(
                        name=exact_match_regex(name) if exact_match_regex_flag else name,
                        **aux_kwargs
                    )
                )
                if res.response and res.response.projects:
                    project_ids.extend([project.id for project in res.response.projects])
                else:
                    projects_not_found.append(name)
            if projects_not_found:
                # If any of the given project names does not exist, fire off a warning
                LoggerRoot.get_base_logger().warning(
                    "No projects were found with name(s): {}".format(", ".join(projects_not_found))
                )
            if not project_ids:
                # If not a single project exists or was found, return empty right away
                return []

        session = cls._get_default_session()
        system_tags = 'system_tags' if hasattr(projects.Project, 'system_tags') else 'tags'
        only_fields = ['id', 'name', 'last_update', system_tags]

        if kwargs and kwargs.get('only_fields'):
            only_fields = list(set(kwargs.pop('only_fields')) | set(only_fields))

        # if we have specific page to look for, we should only get the requested one
        if not fetch_only_first_page and kwargs and 'page' in kwargs:
            fetch_only_first_page = True

        ret_projects = []
        page = -1
        page_size = 500
        while page == -1 or (not fetch_only_first_page and res and len(res.response.projects) == page_size):
            page += 1
            # work on a copy and make sure we override all fields with ours
            request_kwargs = dict(
                id=project_ids,
                project=project_ids if project_ids else kwargs.pop("project", None),
                name=project_name if project_name else kwargs.pop("name", None),
                only_fields=only_fields,
                page=page,
                page_size=page_size,
            )
            # make sure we always override with the kwargs (specifically page selection / page_size)
            request_kwargs.update(kwargs or {})
            res = cls._send(
                session,
                projects.GetAllRequest(**request_kwargs),
            )
            ret_projects.extend(res.response.projects)
        return ret_projects

    @classmethod
    def _wait_for_deferred(cls, project):
        # type: (Optional[Project]) -> None
        """
        Make sure the project object deferred `Project.init` is completed.
        Accessing any of the `project` object's property will ensure the Project.init call was also complete
        This is an internal utility function

        :param project: Optional deferred Project object as returned form Project.init
        """
        if not project:
            return
        # force deferred init to complete
        project.id  # noqa

    @classmethod
    def __get_hash_key(cls, *args):
        def normalize(x):
            return "<{}>".format(x) if x is not None else ""

        return ":".join(map(normalize, args))

    @classmethod
    def __get_last_used_project_id(cls, default_project_name, default_project_name, default_project_type):
        hash_key = cls.__get_hash_key(
            cls._get_api_server(), default_project_name, default_project_name, default_project_type)

        # check if we have a cached project_id we can reuse
        # it must be from within the last 24h and with the same project/name/type
        project_sessions = SessionCache.load_dict(str(cls))

        project_data = project_sessions.get(hash_key)
        if project_data is None:
            return None

        try:
            project_data['type'] = cls.ProjectTypes(project_data['type'])
        except (ValueError, KeyError):
            LoggerRoot.get_base_logger().warning(
                "Corrupted session cache entry: {}. "
                "Unsupported project type: {}"
                "Creating a new project.".format(hash_key, project_data['type']),
            )

            return None

        return project_data

    @classmethod
    def __update_last_used_project_id(cls, default_project_name, default_project_name, default_project_type, project_id):
        hash_key = cls.__get_hash_key(
            cls._get_api_server(), default_project_name, default_project_name, default_project_type)

        project_id = str(project_id)
        # update project session cache
        project_sessions = SessionCache.load_dict(str(cls))
        last_project_session = {'time': time.time(), 'project': default_project_name, 'name': default_project_name,
                             'type': default_project_type, 'id': project_id}

        # remove stale sessions
        for k in list(project_sessions.keys()):
            if ((time.time() - project_sessions[k].get('time', 0)) >
                    60 * 60 * cls.__project_id_reuse_time_window_in_hours):
                project_sessions.pop(k)
        # update current session
        project_sessions[hash_key] = last_project_session
        # store
        SessionCache.store_dict(str(cls), project_sessions)

    @classmethod
    def __project_timed_out(cls, project_data):
        return \
            project_data and \
            project_data.get('id') and \
            project_data.get('time') and \
            (time.time() - project_data.get('time')) > (60 * 60 * cls.__project_id_reuse_time_window_in_hours)

    @classmethod
    def __get_project_api_obj(cls, project_id, only_fields=None):
        if not project_id or cls._offline_mode:
            return None

        all_projects = cls._send(
            cls._get_default_session(),
            projects.GetAllRequest(id=[project_id], only_fields=only_fields),
        ).response.projects

        # The project may not exist in environment changes
        if not all_projects:
            return None

        return all_projects[0]

    @classmethod
    def __project_is_relevant(cls, project_data):
        """
        Check that a cached project is relevant for reuse.

        A project is relevant for reuse if:
            1. It is not timed out i.e it was last use in the previous 24 hours.
            2. It's name, project and type match the data in the server, so not
               to override user changes made by using the UI.

        :param project_data: A mapping from 'id', 'name', 'project', 'type' keys
            to the project's values, as saved in the cache.

        :return: True, if the project is relevant for reuse. False, if not.
        """
        if not project_data:
            return False

        if cls.__project_timed_out(project_data):
            return False

        project_id = project_data.get('id')

        if not project_id:
            return False

        # noinspection PyBroadException
        try:
            project = cls.__get_project_api_obj(project_id, ('id', 'name', 'project', 'type'))
        except Exception:
            project = None

        if project is None:
            return False

        project_name = None
        if project.project:
            # noinspection PyBroadException
            try:
                project = cls._send(
                    cls._get_default_session(),
                    projects.GetByIdRequest(project=project.project)
                ).response.project

                if project:
                    project_name = project.name
            except Exception:
                pass

        if project_data.get('type') and \
                project_data.get('type') not in (cls.ProjectTypes.training, cls.ProjectTypes.testing) and \
                not Session.check_min_api_version(2.8):
            print('WARNING: Changing project type to "{}" : '
                  'clearml-server does not support project type "{}", '
                  'please upgrade clearml-server.'.format(cls.ProjectTypes.training, project_data['type'].value))
            project_data['type'] = cls.ProjectTypes.training

        compares = (
            (project.name, 'name'),
            (project_name, 'project'),
            (project.type, 'type'),
        )

        # compare after casting to string to avoid enum instance issues
        # remember we might have replaced the api version by now, so enums are different
        return all(six.text_type(server_data) == six.text_type(project_data.get(project_data_key))
                   for server_data, project_data_key in compares)

    @classmethod
    def __close_timed_out_project(cls, project_data):
        if not project_data:
            return False

        project = cls.__get_project_api_obj(project_data.get('id'), ('id', 'status'))

        if project is None:
            return False

        stopped_statuses = (
            cls.ProjectStatusEnum.stopped,
            cls.ProjectStatusEnum.published,
            cls.ProjectStatusEnum.publishing,
            cls.ProjectStatusEnum.closed,
            cls.ProjectStatusEnum.failed,
            cls.ProjectStatusEnum.completed,
        )

        if project.status not in stopped_statuses:
            cls._send(
                cls._get_default_session(),
                projects.StoppedRequest(
                    project=project.id,
                    force=True,
                    status_message="Stopped timed out development project"
                ),
            )

            return True
        return False

    @classmethod
    def __add_model_wildcards(cls, auto_connect_frameworks):
        if isinstance(auto_connect_frameworks, dict):
            for k, v in auto_connect_frameworks.items():
                if isinstance(v, str):
                    v = [v]
                if isinstance(v, (list, tuple)):
                    WeightsFileHandler.model_wildcards[k] = [str(i) for i in v]

        def callback(_, model_info):
            if not model_info:
                return None
            parents = Framework.get_framework_parents(model_info.framework)
            wildcards = []
            for parent in parents:
                if WeightsFileHandler.model_wildcards.get(parent):
                    wildcards.extend(WeightsFileHandler.model_wildcards[parent])
            if not wildcards:
                return model_info
            if not matches_any_wildcard(model_info.local_model_path, wildcards):
                return None
            return model_info

        WeightsFileHandler.add_pre_callback(callback)

    def __getstate__(self):
        # type: () -> dict
        return {'main': self.is_main_project(), 'id': self.id, 'offline': self.is_offline()}

    def __setstate__(self, state):
        if state['main'] and not self.__main_project:
            Project.__forked_proc_main_pid = None
            Project.__update_master_pid_project(project=state['id'])
        if state['offline']:
            Project.set_offline(offline_mode=state['offline'])

        project = Project.init(
            continue_last_project=state['id'],
            auto_connect_frameworks={'detect_repository': False}) \
            if state['main'] else Project.get_project(project_id=state['id'])
        self.__dict__ = project.__dict__
