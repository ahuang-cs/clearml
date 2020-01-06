from __future__ import with_statement

import errno
import getpass
import itertools
import json
import os
import shutil
import sys
import threading
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from datetime import datetime
from multiprocessing.pool import ThreadPool
from tempfile import mktemp
from time import time
from types import GeneratorType

import requests
import six
from _socket import gethostname
from attr import attrs, attrib, asdict
from furl import furl
from pathlib2 import Path
from requests.exceptions import ConnectionError
from six import binary_type
from six.moves.queue import Queue, Empty
from six.moves.urllib.parse import urlparse
from six.moves.urllib.request import url2pathname

from ..backend_api.utils import get_http_session_with_retry
from ..backend_config.bucket_config import S3BucketConfigurations, GSBucketConfigurations, AzureContainerConfigurations
from ..config import config
from ..debugging import get_logger
from ..errors import UsageError

log = get_logger('storage')
level = config.get('storage.log.level', None)

if level:
    try:
        log.setLevel(level)
    except (TypeError, ValueError):
        log.error('invalid storage log level in configuration: %s' % level)

upload_pool = ThreadPool(processes=1)


class StorageError(Exception):
    pass


class DownloadError(Exception):
    pass


class _DownloadProgressReport(object):
    def __init__(self, total_size, verbose, remote_path, report_chunk_size_mb, log):
        self._total_size = total_size
        self._verbose = verbose
        self.downloaded_mb = 0.
        self._report_chunk_size = report_chunk_size_mb
        self._log = log
        self.last_reported = 0.
        self._tic = time()
        self._remote_path = remote_path

    def __call__(self, chunk_size):
        chunk_size /= 1024. * 1024.
        self.downloaded_mb += chunk_size
        last_part = self.downloaded_mb - self.last_reported

        if self._verbose or (last_part >= self._report_chunk_size):
            time_diff = time() - self._tic
            speed = (last_part / time_diff) if time_diff != 0 else 0
            self._tic = time()
            self.last_reported = self.downloaded_mb
            self._log.info('Downloading: %.0fMB / %.2fMb @ %.2fMbs from %s' %
                           (self.downloaded_mb, self._total_size, speed, self._remote_path))


@six.add_metaclass(ABCMeta)
class _Driver(object):

    @abstractmethod
    def get_container(self, container_name, config=None, **kwargs):
        pass

    @abstractmethod
    def test_upload(self, test_path, config, **kwargs):
        pass

    @abstractmethod
    def upload_object_via_stream(self, iterator, container, object_name, extra, **kwargs):
        pass

    @abstractmethod
    def list_container_objects(self, container, ex_prefix, **kwargs):
        pass

    @abstractmethod
    def get_direct_access(self, remote_path, **kwargs):
        pass

    @abstractmethod
    def download_object(self, obj, local_path, overwrite_existing, delete_on_failure, callback, **kwargs):
        pass

    @abstractmethod
    def download_object_as_stream(self, obj, chunk_size, **kwargs):
        pass

    @abstractmethod
    def delete_object(self, obj, **kwargs):
        pass

    @abstractmethod
    def upload_object(self, file_path, container, object_name, extra, **kwargs):
        pass

    @abstractmethod
    def get_object(self, container_name, object_name, **kwargs):
        pass


class StorageHelper(object):
    """ Storage helper.
        Used by the entire system to download/upload files.
        Supports both local and remote files (currently local files, network-mapped files, HTTP/S and Amazon S3)
    """
    _temp_download_suffix = '.partially'

    @attrs
    class _PathSubstitutionRule(object):
        registered_prefix = attrib(type=str)
        local_prefix = attrib(type=str)
        replace_windows_sep = attrib(type=bool)
        replace_linux_sep = attrib(type=bool)

        path_substitution_config = 'storage.path_substitution'

        @classmethod
        def load_list_from_config(cls):
            rules_list = []
            for index, sub_config in enumerate(config.get(cls.path_substitution_config, list())):
                rule = cls(
                    registered_prefix=sub_config.get('registered_prefix', None),
                    local_prefix=sub_config.get('local_prefix', None),
                    replace_windows_sep=sub_config.get('replace_windows_sep', False),
                    replace_linux_sep=sub_config.get('replace_linux_sep', False),
                )

                if any(prefix is None for prefix in (rule.registered_prefix, rule.local_prefix)):
                    log.warning(
                        "Illegal substitution rule configuration '{}[{}]': {}".format(
                            cls.path_substitution_config,
                            index,
                            asdict(rule),
                        ))

                    continue

                if all((rule.replace_windows_sep, rule.replace_linux_sep)):
                    log.warning(
                        "Only one of replace_windows_sep and replace_linux_sep flags may be set."
                        "'{}[{}]': {}".format(
                            cls.path_substitution_config,
                            index,
                            asdict(rule),
                        ))
                    continue

                rules_list.append(rule)

            return rules_list

    class _UploadData(object):
        @property
        def src_path(self):
            return self._src_path

        @property
        def dest_path(self):
            return self._dest_path

        @property
        def extra(self):
            return self._extra

        @property
        def callback(self):
            return self._callback

        @property
        def retries(self):
            return self._retries

        def __init__(self, src_path, dest_path, extra, callback, retries):
            self._src_path = src_path
            self._dest_path = dest_path
            self._extra = extra
            self._callback = callback
            self._retries = retries

        def __str__(self):
            return "src=%s" % self.src_path

    _helpers = {}  # cache of helper instances

    # global terminate event for async upload threads
    _terminate = threading.Event()
    _async_upload_threads = set()

    # collect all bucket credentials that aren't empty (ignore entries with an empty key or secret)
    _s3_configurations = S3BucketConfigurations.from_config(config.get('aws.s3', {}))
    _gs_configurations = GSBucketConfigurations.from_config(config.get('google.storage', {}))
    _azure_configurations = AzureContainerConfigurations.from_config(config.get('azure.storage', {}))

    _path_substitutions = _PathSubstitutionRule.load_list_from_config()

    @property
    def log(self):
        return self._log

    @property
    def scheme(self):
        return self._scheme

    @property
    def secure(self):
        return self._secure

    @property
    def base_url(self):
        return self._base_url

    @classmethod
    def get(cls, url, logger=None, **kwargs):
        """
        Get a storage helper instance for the given URL

        :return: StorageHelper instance
        """

        # Handle URL substitution etc before locating the correct storage driver
        url = cls._canonize_url(url)

        # Get the credentials we should use for this url
        base_url = cls._resolve_base_url(url)

        instance_key = '%s_%s' % (base_url, threading.current_thread().ident or 0)

        force_create = kwargs.pop('__force_create', False)
        if (instance_key in cls._helpers) and (not force_create):
            return cls._helpers[instance_key]

        # Don't canonize URL since we already did it
        try:
            instance = cls(base_url=base_url, url=url, logger=logger, canonize_url=False, **kwargs)
        except Exception:
            log.error("Failed credentials for {}".format(base_url or url))
            return None

        cls._helpers[instance_key] = instance
        return instance

    @classmethod
    def get_local_copy(cls, remote_url):
        """
        Download a file from remote URL to a local storage, and return path to local copy,

        :param remote_url: Remote URL. Example: https://example.com/file.jpg s3://bucket/folder/file.mp4 etc.
        :return: Path to local copy of the downloaded file. None if error occurred.
        """
        helper = cls.get(remote_url)
        if not helper:
            return None
        # create temp file with the requested file name
        file_name = '.' + remote_url.split('/')[-1].split(os.path.sep)[-1]
        local_path = mktemp(suffix=file_name)
        return helper.download_to_file(remote_url, local_path)

    def __init__(self, base_url, url, key=None, secret=None, region=None, verbose=False, logger=None, retries=5,
                 **kwargs):
        self._log = logger or log
        self._verbose = verbose
        self._retries = retries
        self._extra = {}
        self._base_url = base_url
        self._secure = True
        self._driver = None
        self._container = None
        self._conf = None

        if kwargs.get('canonize_url', True):
            url = self._canonize_url(url)

        parsed = urlparse(url)
        self._scheme = parsed.scheme

        if self._scheme == _AzureBlobServiceStorageDriver.scheme:
            self._conf = copy(self._azure_configurations.get_config_by_uri(url))
            if self._conf is None:
                raise StorageError("Missing Azure Blob Storage configuration for {}".format(url))

            if not self._conf.account_name or not self._conf.account_key:
                raise StorageError(
                    "Missing account name or key for Azure Blob Storage access for {}".format(base_url)
                )

            self._driver = _AzureBlobServiceStorageDriver()
            self._container = self._driver.get_container(config=self._conf)

        elif self._scheme == _Boto3Driver.scheme:
            self._conf = copy(self._s3_configurations.get_config_by_uri(url))
            self._secure = self._conf.secure

            final_region = region if region else self._conf.region
            if not final_region:
                final_region = None

            self._conf.update(
                key=key or self._conf.key,
                secret=secret or self._conf.secret,
                multipart=self._conf.multipart,
                region=final_region,
            )

            if not self._conf.key or not self._conf.secret:
                raise ValueError('Missing key and secret for S3 storage access (%s)' % base_url)

            self._driver = _Boto3Driver()
            self._container = self._driver.get_container(container_name=self._base_url, retries=retries,
                                                         config=self._conf)

        elif self._scheme == _GoogleCloudStorageDriver.scheme:
            self._conf = copy(self._gs_configurations.get_config_by_uri(url))
            self._driver = _GoogleCloudStorageDriver()
            self._container = self._driver.get_container(
                container_name=self._base_url,
                config=self._conf
            )

        elif self._scheme in _HttpDriver.schemes:
            self._driver = _HttpDriver(retries=retries)
            self._container = self._driver.get_container(container_name=self._base_url)
        else:  # elif self._scheme == 'file':
            # if this is not a known scheme assume local file

            # If the scheme is file, use only the path segment, If not, use the entire URL
            if self._scheme == 'file':
                url = parsed.path

            url = url.replace("\\", "/")

            # url2pathname is specifically intended to operate on (urlparse result).path
            # and returns a cross-platform compatible result
            driver_uri = url2pathname(url)
            if Path(driver_uri).is_file():
                driver_uri = str(Path(driver_uri).parent)
            elif not Path(driver_uri).exists():
                # assume a folder and create
                Path(driver_uri).mkdir(parents=True, exist_ok=True)

            self._driver = _FileStorageDriver(driver_uri)
            self._container = self._driver.get_container(container_name='.')

    @classmethod
    def terminate_uploads(cls, force=True, timeout=2.0):
        if force:
            # since async uploaders are daemon threads, we can just return and let them close by themselves
            return
        # signal all threads to terminate and give them a chance for 'timeout' seconds (total, not per-thread)
        cls._terminate.set()
        remaining_timeout = timeout
        for thread in cls._async_upload_threads:
            t = time()
            try:
                thread.join(timeout=remaining_timeout)
            except Exception:
                pass
            remaining_timeout -= (time() - t)

    @classmethod
    def get_configuration(cls, bucket_config):
        return cls._s3_configurations.get_config_by_bucket(bucket_config.bucket, bucket_config.host)

    @classmethod
    def add_configuration(cls, bucket_config, log=None, _test_config=True):
        # Try to use existing configuration if we have no key and secret
        use_existing = not bucket_config.is_valid()

        # Get existing config anyway (we'll either try to use it or alert we're replacing it
        existing = cls.get_configuration(bucket_config)

        configs = cls._s3_configurations

        if not use_existing:
            # Test bucket config, fails if unsuccessful
            if _test_config:
                _Boto3Driver._test_bucket_config(bucket_config, log)

            if existing:
                if log:
                    log.warning('Overriding existing configuration for %s/%s'
                                % (existing.host or 'AWS', existing.bucket))
                configs.remove_config(existing)
        else:
            # Try to use existing configuration
            good_config = False
            if existing:
                if log:
                    log.info('Using existing credentials for bucket %s/%s'
                             % (bucket_config.host or 'AWS', bucket_config.bucket))
                good_config = _Boto3Driver._test_bucket_config(existing, log, raise_on_error=False)

            if not good_config:
                # Try to use global key/secret
                configs.update_config_with_defaults(bucket_config)

                if log:
                    log.info('Using global credentials for bucket %s/%s'
                             % (bucket_config.host or 'AWS', bucket_config.bucket))
                if _test_config:
                    _Boto3Driver._test_bucket_config(bucket_config, log)
            else:
                # do not add anything, existing config is OK
                return

        configs.add_config(bucket_config)

    @classmethod
    def add_path_substitution(
            cls,
            registered_prefix,
            local_prefix,
            replace_windows_sep=False,
            replace_linux_sep=False,
    ):
        """
        Add a path substitution rule for storage paths.

        Useful for case where the data was registered under some path, and that
        path was later renamed. This may happen with local storage paths where
        each machine is has different mounts or network drives configurations

        :param registered_prefix: The prefix to search for and replace. This is
            the prefix of the path the data is registered under. This should be the
            exact url prefix, case sensitive, as the data is registered.
        :param local_prefix: The prefix to replace 'registered_prefix' with. This
            is the prefix of the path the data is actually saved under. This should be the
            exact url prefix, case sensitive, as the data is saved under.
        :param replace_windows_sep: If set to True, and the prefix matches, the rest
            of the url has all of the windows path separators (backslash '\') replaced with
            the native os path separator.
        :param replace_linux_sep: If set to True, and the prefix matches, the rest
            of the url has all of the linux/unix path separators (slash '/') replaced with
            the native os path separator.
        """

        if not registered_prefix or not local_prefix:
            raise UsageError("Path substitution prefixes must be non empty strings")

        if replace_windows_sep and replace_linux_sep:
            raise UsageError("Only one of replace_windows_sep and replace_linux_sep may be set.")

        rule = cls._PathSubstitutionRule(
            registered_prefix=registered_prefix,
            local_prefix=local_prefix,
            replace_windows_sep=replace_windows_sep,
            replace_linux_sep=replace_linux_sep,
        )

        cls._path_substitutions.append(rule)

    @classmethod
    def clear_path_substitutions(cls):
        """
        Removes all path substitution rules, including ones from the configuration file.
        """
        cls._path_substitutions = list()

    def verify_upload(self, folder_uri='', raise_on_error=True, log_on_error=True):
        """
        Verify that this helper can upload files to a folder.

        An upload is possible iff:
            1. the destination folder is under the base uri of the url used to create the helper
            2. the helper has credentials to write to the destination folder

        :param folder_uri: The destination folder to test. Must be an absolute
            url that begins with the base uri of the url used to create the helper.
        :param raise_on_error: Raise an exception if an upload is not possible
        :param log_on_error: Log an error if an upload is not possible
        :return: True iff an upload to folder_uri is possible.
        """

        folder_uri = self._canonize_url(folder_uri)

        folder_uri = self.conform_url(folder_uri, self._base_url)

        test_path = self._normalize_object_name(folder_uri)

        if self._scheme == _Boto3Driver.scheme:
            _Boto3Driver._test_bucket_config(
                self._conf,
                self._log,
                test_path=test_path,
                raise_on_error=raise_on_error,
                log_on_error=log_on_error,
            )
        elif self._scheme == _GoogleCloudStorageDriver.scheme:
            self._driver.test_upload(test_path, self._conf)

        elif self._scheme == 'file':
            # Check path exists
            Path(test_path).mkdir(parents=True, exist_ok=True)
            # check path permissions
            Path(test_path).touch(exist_ok=True)

        return folder_uri

    def upload_from_stream(self, stream, dest_path, extra=None, retries=1):
        dest_path = self._canonize_url(dest_path)
        object_name = self._normalize_object_name(dest_path)
        extra = extra.copy() if extra else {}
        extra.update(self._extra)
        last_ex = None
        for i in range(max(1, retries)):
            try:
                self._driver.upload_object_via_stream(
                    iterator=stream,
                    container=self._container,
                    object_name=object_name,
                    extra=extra)
                last_ex = None
                break
            except Exception as ex:
                last_ex = ex
                # seek to beginning if possible
                try:
                    stream.seek(0)
                except:
                    pass
        if last_ex:
            raise last_ex

        return dest_path

    def upload(self, src_path, dest_path=None, extra=None, async_enable=False, cb=None, retries=1):
        if not dest_path:
            dest_path = os.path.basename(src_path)

        dest_path = self._canonize_url(dest_path)

        if async_enable:
            data = self._UploadData(src_path=src_path, dest_path=dest_path, extra=extra, callback=cb, retries=retries)
            return upload_pool.apply_async(self._do_async_upload, args=(data,))
        else:
            return self._do_upload(src_path, dest_path, extra, cb, verbose=False, retries=retries)

    def list(self, prefix=None):
        """
        List entries in the helper base path.
        
        Return a list of names inside this helper base path. The base path is
        determined at creation time and is specific for each storage medium.
        For Google Storage and S3 it is the bucket of the path.
        For local files it is the root directory.
        
        This operation is not supported for http and https protocols.
        
        :param prefix: If None, return the list as described above. If not, it
            must be a string - the path of a sub directory under the base path.
            the returned list will include only objects under that subdir.
            
        :return: List of strings - the paths of all the objects in the storage base
            path under prefix. Listed relative to the base path.
            
        """
        
        if prefix:
            if prefix.startswith(self._base_url):
                prefix = prefix[len(self.base_url):].lstrip("/")
                
            try:
                res = self._driver.list_container_objects(self._container, ex_prefix=prefix)
            except TypeError:
                res = self._driver.list_container_objects(self._container)

            return [
                obj.name
                for obj in res if
                obj.name.startswith(prefix) and obj.name != prefix
            ]
        else:
            return [obj.name for obj in self._driver.list_container_objects(self._container)]

    def download_to_file(self, remote_path, local_path, overwrite_existing=False, delete_on_failure=True, verbose=None):
        def next_chunk(astream):
            _tic = time()
            if isinstance(astream, binary_type):
                chunk = astream
                astream = None
            elif astream:
                try:
                    chunk = next(astream)
                except StopIteration:
                    chunk = None
            else:
                chunk = None
            _tic = time() - _tic
            return chunk, astream, _tic

        remote_path = self._canonize_url(remote_path)
        verbose = self._verbose if verbose is None else verbose

        # Check if driver type supports direct access:
        direct_access_path = self._driver.get_direct_access(remote_path)
        if direct_access_path:
            return direct_access_path

        temp_local_path = None
        try:
            if verbose:
                self._log.info('Start downloading from %s' % remote_path)
            if not overwrite_existing and Path(local_path).is_file():
                self._log.warning(
                    'File {} already exists, no need to download, thread id = {}'.format(
                        local_path,
                        threading.current_thread().ident,
                    ),
                )

                return local_path
            # we download into temp_local_path so that if we accidentally stop in the middle,
            # we won't think we have the entire file
            temp_local_path = '{}_{}{}'.format(local_path, time(), self._temp_download_suffix)
            obj = self._get_object(remote_path)
            if not obj:
                return None

            # object size in bytes
            total_size_mb = -1
            dl_total_mb = 0.
            download_reported = False
            # chunks size is ignored and always 5Mb
            chunk_size_mb = 5

            # try to get file size
            try:
                if isinstance(self._driver, _HttpDriver) and obj:
                    total_size_mb = float(obj.headers.get('Content-Length', 0)) / (1024 * 1024)
                elif hasattr(obj, 'size'):
                    size = obj.size
                    # Google storage has the option to reload the object to get the size
                    if size is None and hasattr(obj, 'reload'):
                        obj.reload()
                        size = obj.size

                    total_size_mb = 0 if size is None else float(size) / (1024 * 1024)
                elif hasattr(obj, 'content_length'):
                    total_size_mb = float(obj.content_length) / (1024 * 1024)
            except (ValueError, AttributeError, KeyError):
                pass

            # if driver supports download with call back, use it (it might be faster)
            if hasattr(self._driver, 'download_object'):
                # callback
                cb = _DownloadProgressReport(total_size_mb, verbose,
                                             remote_path, chunk_size_mb, self._log)
                self._driver.download_object(obj, temp_local_path, callback=cb)
                download_reported = bool(cb.last_reported)
                dl_total_mb = cb.downloaded_mb
            else:
                stream = self._driver.download_object_as_stream(obj, chunk_size_mb * 1024 * 1024)
                if stream is None:
                    raise ValueError('Could not download %s' % remote_path)
                with open(temp_local_path, 'wb') as fd:
                    data, stream, tic = next_chunk(stream)
                    while data:
                        fd.write(data)
                        dl_rate = len(data) / float(1024 * 1024 * tic + 0.000001)
                        dl_total_mb += len(data) / float(1024 * 1024)
                        # report download if we are on the second chunk
                        if verbose or (dl_total_mb * 0.9 > chunk_size_mb):
                            download_reported = True
                            self._log.info('Downloading: %.0fMB / %.2fMb @ %.2fMbs from %s' %
                                           (dl_total_mb, total_size_mb, dl_rate, remote_path))
                        data, stream, tic = next_chunk(stream)

            if Path(temp_local_path).stat().st_size <= 0:
                raise Exception('downloaded a 0-sized file')

            # if we are on windows, we need to remove the target file before renaming
            # otherwise posix rename will overwrite the target
            if os.name != 'posix':
                try:
                    os.remove(local_path)
                except Exception:
                    pass

            # rename temp file to local_file
            os.rename(temp_local_path, local_path)
            # report download if we are on the second chunk
            if verbose or download_reported:
                self._log.info(
                    'Downloaded %.2f MB successfully from %s , saved to %s' % (dl_total_mb, remote_path, local_path))
            return local_path
        except DownloadError as e:
            raise
        except Exception as e:
            self._log.error("Could not download {} , err: {} ".format(remote_path, e))
            if delete_on_failure:
                try:
                    if temp_local_path:
                        os.remove(temp_local_path)
                except:
                    pass
            return None

    def download_as_stream(self, remote_path, chunk_size=None):
        remote_path = self._canonize_url(remote_path)
        try:
            obj = self._get_object(remote_path)
            return self._driver.download_object_as_stream(obj, chunk_size=chunk_size)
        except DownloadError as e:
            raise
        except Exception as e:
            self._log.error("Could not download file : %s, err:%s " % (remote_path, str(e)))
            return None

    def download_as_nparray(self, remote_path, chunk_size=None):
        try:
            stream = self.download_as_stream(remote_path, chunk_size)
            if stream is None:
                return

            # TODO: ugly py3 hack, please remove ASAP
            if six.PY3 and not isinstance(stream, GeneratorType):
                import numpy as np
                return np.frombuffer(stream, dtype=np.uint8)
            else:
                import numpy as np
                return np.asarray(bytearray(b''.join(stream)), dtype=np.uint8)

        except Exception as e:
            self._log.error("Could not download file : %s, err:%s " % (remote_path, str(e)))

    def delete(self, path):
        return self._driver.delete_object(self._get_object(path))

    def check_write_permissions(self, dest_path=None):
        # create a temporary file, then de;ete it
        base_url = dest_path or self._base_url
        dest_path = base_url + '/.trains.test'
        # do not check http/s connection permissions
        if dest_path.startswith('http'):
            return True
        try:
            self.upload_from_stream(stream=six.BytesIO(b'trains'), dest_path=dest_path)
            self.delete(path=dest_path)
        except Exception:
            raise ValueError('Insufficient permissions for {}'.format(base_url))
        return True

    @classmethod
    def download_from_url(cls, remote_url, local_path, overwrite_existing=False):
        """
        Download a file from remote URL to a local storage

        :param remote_url: Remote URL. Example: https://example.com/image.jpg or s3://bucket/folder/file.mp4 etc.
        :param local_path: target location for downloaded file. Example: /tmp/image.jpg
        :param overwrite_existing: If True and local_path exists, it will overwrite it, otherwise print warning
        :return: local_path if download was successful
        """
        helper = cls.get(remote_url)
        if not helper:
            return None
        return helper.download_to_file(remote_url, local_path, overwrite_existing=overwrite_existing)

    @classmethod
    def _canonize_url(cls, url):
        return cls._apply_url_substitutions(url)

    @classmethod
    def _apply_url_substitutions(cls, url):
        def replace_separator(_url, where, sep):
            return _url[:where] + _url[where:].replace(sep, os.sep)

        for index, rule in enumerate(cls._path_substitutions):
            if url.startswith(rule.registered_prefix):
                url = url.replace(
                    rule.registered_prefix,
                    rule.local_prefix,
                    1,  # count. str.replace() does not support keyword arguments
                )

                if rule.replace_windows_sep:
                    url = replace_separator(url, len(rule.local_prefix), '\\')

                if rule.replace_linux_sep:
                    url = replace_separator(url, len(rule.local_prefix), '/')

                break

        return url

    @classmethod
    def _resolve_base_url(cls, base_url):
        parsed = urlparse(base_url)
        if parsed.scheme == _Boto3Driver.scheme:
            conf = cls._s3_configurations.get_config_by_uri(base_url)
            bucket = conf.bucket
            if not bucket:
                parts = Path(parsed.path.strip('/')).parts
                if parts:
                    bucket = parts[0]
            return '/'.join(x for x in ('s3:/', conf.host, bucket) if x)
        elif parsed.scheme == _AzureBlobServiceStorageDriver.scheme:
            conf = cls._azure_configurations.get_config_by_uri(base_url)
            if not conf:
                raise StorageError("Can't find azure configuration for {}".format(base_url))
            return str(furl(base_url).set(path=conf.container_name))
        elif parsed.scheme == _GoogleCloudStorageDriver.scheme:
            conf = cls._gs_configurations.get_config_by_uri(base_url)
            return str(furl(scheme=parsed.scheme, netloc=conf.bucket))
        elif parsed.scheme == 'http':
            return 'http://'
        elif parsed.scheme == 'https':
            return 'https://'
        else:  # if parsed.scheme == 'file':
            # if we do not know what it is, we assume file
            return 'file://'

    @classmethod
    def conform_url(cls, folder_uri, base_url=None):
        if not folder_uri:
            return folder_uri
        _base_url = cls._resolve_base_url(folder_uri) if not base_url else base_url

        if not folder_uri.startswith(_base_url):
            prev_folder_uri = folder_uri
            if _base_url == 'file://':
                folder_uri = str(Path(folder_uri).absolute())
                if folder_uri.startswith('/'):
                    folder_uri = _base_url + folder_uri
                else:
                    folder_uri = '/'.join((_base_url, folder_uri))

                log.debug('Upload destination {} amended to {} for registration purposes'.format(
                    prev_folder_uri, folder_uri))
            else:
                raise ValueError('folder_uri: {} does not start with base url: {}'.format(folder_uri, _base_url))

        return folder_uri

    def _absolute_object_name(self, path):
        """ Returns absolute remote path, including any prefix that is handled by the container """
        if not path.startswith(self.base_url):
            return self.base_url.rstrip('/') + '///' + path.lstrip('/')
        return path

    def _normalize_object_name(self, path):
        """ Normalize remote path. Remove any prefix that is already handled by the container """
        if path.startswith(self.base_url):
            path = path[len(self.base_url):]
            if path.startswith('/') and os.name == 'nt':
                path = path[1:]
        if self.scheme in (_Boto3Driver.scheme, _GoogleCloudStorageDriver.scheme,
                           _AzureBlobServiceStorageDriver.scheme):
            path = path.lstrip('/')
        return path

    def _do_async_upload(self, data):
        assert isinstance(data, self._UploadData)
        return self._do_upload(data.src_path, data.dest_path, extra=data.extra, cb=data.callback,
                               verbose=True, retries=data.retries)

    def _upload_from_file(self, local_path, dest_path, extra=None):
        if not hasattr(self._driver, 'upload_object'):
            with open(local_path, 'rb') as stream:
                res = self.upload_from_stream(stream=stream, dest_path=dest_path, extra=extra)
        else:
            object_name = self._normalize_object_name(dest_path)
            extra = extra.copy() if extra else {}
            extra.update(self._extra)
            res = self._driver.upload_object(
                file_path=local_path,
                container=self._container,
                object_name=object_name,
                extra=extra)
        return res

    def _do_upload(self, src_path, dest_path, extra=None, cb=None, verbose=False, retries=1):
        object_name = self._normalize_object_name(dest_path)
        if cb:
            try:
                cb(None)
            except Exception as e:
                self._log.error("Calling upload callback when starting upload: %s" % str(e))
        if verbose:
            msg = "Starting upload: %s => %s" % (src_path, object_name)
            if object_name.startswith('file://') or object_name.startswith('/'):
                self._log.debug(msg)
            else:
                self._log.info(msg)
        last_ex = None
        for i in range(max(1, retries)):
            try:
                self._upload_from_file(local_path=src_path, dest_path=dest_path, extra=extra)
                last_ex = None
                break
            except Exception as e:
                last_ex = e

        if last_ex:
            self._log.error("Exception encountered while uploading %s" % str(last_ex))
            try:
                cb(False)
            except Exception as e:
                self._log.warning("Exception on upload callback: %s" % str(e))
            raise last_ex

        if verbose:
            self._log.debug("Finished upload: %s => %s" % (src_path, object_name))
        if cb:
            try:
                cb(dest_path)
            except Exception as e:
                self._log.warning("Exception on upload callback: %s" % str(e))
        
        return dest_path

    def _get_object(self, path):
        object_name = self._normalize_object_name(path)
        try:
            return self._driver.get_object(container_name=self._container.name, object_name=object_name)
        except ConnectionError as ex:
            raise DownloadError
        except Exception as e:
            self.log.warning('Storage helper problem for {}: {}'.format(str(object_name), str(e)))
            return None


class _HttpDriver(_Driver):
    """ LibCloud http/https adapter (simple, enough for now) """

    timeout = (5.0, 30.)
    min_kbps_speed = 50

    schemes = ('http', 'https')

    class _Container(object):
        _default_backend_session = None
        _default_files_server_host = None

        def __init__(self, name, retries=5, **kwargs):
            self.name = name
            self.session = get_http_session_with_retry(total=retries, connect=retries, read=retries, redirect=retries)

        def get_headers(self, url):
            if not self._default_backend_session:
                from ..backend_interface.base import InterfaceBase
                self._default_backend_session = InterfaceBase._get_default_session()
            if self._default_files_server_host is None:
                self._default_files_server_host = self._default_backend_session.get_files_server_host().rstrip('/')

            if url == self._default_files_server_host or url.startswith(self._default_files_server_host + '/'):
                return self._default_backend_session.add_auth_headers({})
            return None

    def __init__(self, retries=5):
        self._retries = retries
        self._containers = {}

    def get_container(self, container_name, config=None, **kwargs):
        if container_name not in self._containers:
            self._containers[container_name] = self._Container(name=container_name, retries=self._retries, **kwargs)
        return self._containers[container_name]

    def upload_object_via_stream(self, iterator, container, object_name, extra=None, **kwargs):
        url = object_name[:object_name.index('/')]
        url_path = object_name[len(url)+1:]
        full_url = container.name+url
        # when sending data in post, there is no connection timeout, just an entire upload timeout
        timeout = self.timeout[-1]
        if hasattr(iterator, 'tell') and hasattr(iterator, 'seek'):
            pos = iterator.tell()
            iterator.seek(0, 2)
            stream_size = iterator.tell() - pos
            iterator.seek(pos, 0)
            timeout = max(timeout, (stream_size / 1024) / float(self.min_kbps_speed))

        res = container.session.post(full_url, files={url_path: iterator}, timeout=timeout,
                                     headers=container.get_headers(full_url))
        if res.status_code != requests.codes.ok:
            raise ValueError('Failed uploading object %s (%d): %s' % (object_name, res.status_code, res.text))
        return res

    def list_container_objects(self, *args, **kwargs):
        raise NotImplementedError('List is not implemented for http protocol')

    def delete_object(self, *args, **kwargs):
        raise NotImplementedError('Delete is not implemented for http protocol')

    def get_object(self, container_name, object_name, *args, **kwargs):
        container = self._containers[container_name]
        # set stream flag before get request
        container.session.stream = kwargs.get('stream', True)
        url = ''.join((container_name, object_name.lstrip('/')))
        res = container.session.get(url, timeout=self.timeout, headers=container.get_headers(url))
        if res.status_code != requests.codes.ok:
            raise ValueError('Failed getting object %s (%d): %s' % (object_name, res.status_code, res.text))
        return res

    def download_object_as_stream(self, obj, chunk_size=64 * 1024, **_):
        # return iterable object
        return obj.iter_content(chunk_size=chunk_size)

    def download_object(self, obj, local_path, overwrite_existing=True, delete_on_failure=True, callback=None, **_):
        p = Path(local_path)
        if not overwrite_existing and p.is_file():
            log.warning('failed saving after download: overwrite=False and file exists (%s)' % str(p))
            return
        length = 0
        with p.open(mode='wb') as f:
            for chunk in obj.iter_content(chunk_size=5 * 1024 * 1024):
                # filter out keep-alive new chunks
                if not chunk:
                    continue
                chunk_size = len(chunk)
                f.write(chunk)
                length += chunk_size
                if callback:
                    callback(chunk_size)

        return length

    def get_direct_access(self, remote_path, **_):
        return None

    def test_upload(self, test_path, config, **kwargs):
        return True

    def upload_object(self, file_path, container, object_name, extra, **kwargs):
        with open(file_path, 'rb') as stream:
            return self.upload_object_via_stream(iterator=stream, container=container,
                                                 object_name=object_name, extra=extra, **kwargs)


class _Stream(object):
    encoding = None
    mode = 'rw'
    name = ''
    newlines = '\n'
    softspace = False

    def __init__(self, input_iterator=None):
        self.closed = False
        self._buffer = Queue()
        self._input_iterator = input_iterator
        self._leftover = None

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def close(self):
        self.closed = True

    def flush(self):
        pass

    def fileno(self):
        return 87

    def isatty(self):
        return False

    def next(self):
        while not self.closed or not self._buffer.empty():
            # input stream
            if self._input_iterator:
                try:
                    chunck = next(self._input_iterator)
                    return chunck
                except StopIteration:
                    self.closed = True
                    raise StopIteration()
                except Exception as ex:
                    log.error('Failed downloading: %s' % ex)
            else:
                # in/out stream
                try:
                    return self._buffer.get(block=True, timeout=1.)
                except Empty:
                    pass

        raise StopIteration()

    def read(self, size=None):
        try:
            data = self.next() if self._leftover is None else self._leftover
        except StopIteration:
            return six.b('')

        self._leftover = None
        try:
            while size is None or not data or len(data) < size:
                chunk = self.next()
                if chunk is not None:
                    if data is not None:
                        data += chunk
                    else:
                        data = chunk
        except StopIteration:
            pass

        if size is not None and data and len(data) > size:
            self._leftover = data[size:]
            return data[:size]

        return data

    def readline(self, size=None):
        return self.read(size)

    def readlines(self, sizehint=None):
        pass

    def truncate(self, size=None):
        pass

    def write(self, bytes):
        self._buffer.put(bytes, block=True)

    def writelines(self, sequence):
        for s in sequence:
            self.write(s)


class _Boto3Driver(_Driver):
    """ Boto3 storage adapter (simple, enough for now) """

    _max_multipart_concurrency = config.get('aws.boto3.max_multipart_concurrency', 16)

    _min_pool_connections = 512
    _pool_connections = config.get('aws.boto3.pool_connections', 512)

    _stream_download_pool_connections = 128
    _stream_download_pool = None

    _containers = {}

    scheme = 's3'
    scheme_prefix = str(furl(scheme=scheme, netloc=''))

    _bucket_location_failure_reported = set()

    class _Container(object):
        _creation_lock = threading.Lock()

        def __init__(self, name, cfg):
            try:
                import boto3
                import botocore.client
                from botocore.exceptions import ClientError
            except ImportError:
                raise UsageError(
                    'AWS S3 storage driver (boto3) not found. '
                    'Please install driver using "pip install \'boto3>=1.9\'"'
                )

            # skip 's3://'
            self.name = name[5:]
            endpoint = (('https://' if cfg.secure else 'http://') + cfg.host) if cfg.host else None

            # boto3 client creation isn't thread-safe (client itself is)
            with self._creation_lock:
                self.resource = boto3.resource(
                    's3',
                    aws_access_key_id=cfg.key,
                    aws_secret_access_key=cfg.secret,
                    endpoint_url=endpoint,
                    use_ssl=cfg.secure,
                    config=botocore.client.Config(
                        max_pool_connections=max(
                            _Boto3Driver._min_pool_connections,
                            _Boto3Driver._pool_connections)
                    ),
                )

                self.config = cfg
                bucket_name = self.name[len(cfg.host) + 1:] if cfg.host else self.name
                self.bucket = self.resource.Bucket(bucket_name)

    @attrs
    class ListResult(object):
        name = attrib(default=None)

    def __init__(self):
        pass

    def _get_stream_download_pool(self):
        if self._stream_download_pool is None:
            self._stream_download_pool = ThreadPoolExecutor(max_workers=self._stream_download_pool_connections)
        return self._stream_download_pool

    def get_container(self, container_name, config=None, **kwargs):
        if container_name not in self._containers:
            self._containers[container_name] = self._Container(name=container_name, cfg=config)
        self._containers[container_name].config.retries = kwargs.get('retries', 5)
        return self._containers[container_name]

    def upload_object_via_stream(self, iterator, container, object_name, extra=None, **kwargs):
        import boto3.s3.transfer
        stream = _Stream(iterator)
        try:
            container.bucket.upload_fileobj(stream, object_name, Config=boto3.s3.transfer.TransferConfig(
                use_threads=container.config.multipart,
                max_concurrency=self._max_multipart_concurrency if container.config.multipart else 1,
                num_download_attempts=container.config.retries))
        except Exception as ex:
            log.error('Failed uploading: %s' % ex)
            return False
        return True

    def upload_object(self, file_path, container, object_name, extra=None, **kwargs):
        import boto3.s3.transfer
        try:
            container.bucket.upload_file(file_path, object_name, Config=boto3.s3.transfer.TransferConfig(
                use_threads=container.config.multipart,
                max_concurrency=self._max_multipart_concurrency if container.config.multipart else 1,
                num_download_attempts=container.config.retries))
        except Exception as ex:
            log.error('Failed uploading: %s' % ex)
            return False
        return True

    def list_container_objects(self, container, ex_prefix=None, **kwargs):
        if ex_prefix:
            res = container.bucket.objects.filter(Prefix=ex_prefix)
        else:
            res = container.bucket.objects.all()
        for res in res:
            yield self.ListResult(name=res.key)

    def delete_object(self, object, **kwargs):
        object.delete()

    def get_object(self, container_name, object_name, *args, **kwargs):
        full_container_name = 's3://' + container_name
        container = self._containers[full_container_name]
        obj = container.resource.Object(container.bucket.name, object_name)
        obj.container_name = full_container_name
        return obj

    def download_object_as_stream(self, obj, chunk_size=64 * 1024, **_):
        def async_download(a_obj, a_stream, cfg):
            try:
                a_obj.download_fileobj(a_stream, Config=cfg)
            except Exception as ex:
                log.error('Failed downloading: %s' % ex)
            a_stream.close()

        import boto3.s3.transfer
        # return iterable object
        stream = _Stream()
        container = self._containers[obj.container_name]
        config = boto3.s3.transfer.TransferConfig(
            use_threads=container.config.multipart,
            max_concurrency=self._max_multipart_concurrency if container.config.multipart else 1,
            num_download_attempts=container.config.retries)
        self._get_stream_download_pool().submit(async_download, obj, stream, config)

        return stream

    def download_object(self, obj, local_path, overwrite_existing=True, delete_on_failure=True, callback=None, **_):
        import boto3.s3.transfer
        p = Path(local_path)
        if not overwrite_existing and p.is_file():
            log.warning('failed saving after download: overwrite=False and file exists (%s)' % str(p))
            return
        container = self._containers[obj.container_name]
        obj.download_file(str(p),
                          Callback=callback,
                          Config=boto3.s3.transfer.TransferConfig(
                              use_threads=container.config.multipart,
                              max_concurrency=self._max_multipart_concurrency if container.config.multipart else 1,
                              num_download_attempts=container.config.retries))

    @classmethod
    def _test_bucket_config(cls, conf, log, test_path='', raise_on_error=True, log_on_error=True):
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            return False

        if not conf.bucket:
            return False
        try:
            if not conf.is_valid():
                raise Exception('Missing credentials')

            fullname = furl(conf.bucket).add(path=test_path).add(path='%s-upload_test' % cls.__module__)
            bucket_name = str(fullname.path.segments[0])
            filename = str(furl(path=fullname.path.segments[1:]))

            data = {
                'user': getpass.getuser(),
                'machine': gethostname(),
                'time': datetime.utcnow().isoformat()
            }

            boto_session = boto3.Session(conf.key, conf.secret)
            boto_resource = boto_session.resource('s3', conf.region)
            bucket = boto_resource.Bucket(bucket_name)
            bucket.put_object(Key=filename, Body=six.b(json.dumps(data)))

            region = cls._get_bucket_region(conf=conf, log=log, report_info=True)

            if region and ((conf.region and region != conf.region) or (not conf.region and region != 'us-east-1')):
                msg = "incorrect region specified for bucket %s (detected region %s)" % (conf.bucket, region)
            else:
                return True

        except ClientError as ex:
            msg = ex.response['Error']['Message']
            if log_on_error and log:
                log.error(msg)

            if raise_on_error:
                raise

        except Exception as ex:
            msg = str(ex)
            if log_on_error and log:
                log.error(msg)

            if raise_on_error:
                raise

        msg = ("Failed testing access to bucket %s: " % conf.bucket) + msg

        if log_on_error and log:
            log.error(msg)

        if raise_on_error:
            raise StorageError(msg)

        return False

    @classmethod
    def _get_bucket_region(cls, conf, log=None, report_info=False):
        import boto3
        from botocore.exceptions import ClientError

        if not conf.bucket:
            return None

        def report(msg):
            if log and conf.get_bucket_host() not in cls._bucket_location_failure_reported:
                if report_info:
                    log.debug(msg)
                else:
                    log.warning(msg)
                cls._bucket_location_failure_reported.add(conf.get_bucket_host())

        try:
            boto_session = boto3.Session(conf.key, conf.secret)
            boto_resource = boto_session.resource('s3')
            return boto_resource.meta.client.get_bucket_location(Bucket=conf.bucket)["LocationConstraint"]

        except ClientError as ex:
            report("Failed getting bucket location (region) for bucket "
                   "%s: %s (%s, access_key=%s). Default region will be used. "
                   "This is normal if you do not have GET_BUCKET_LOCATION permission"
                   % (conf.bucket, ex.response['Error']['Message'], ex.response['Error']['Code'], conf.key))
        except Exception as ex:
            report("Failed getting bucket location (region) for bucket %s: %s. Default region will be used."
                   % (conf.bucket, str(ex)))

        return None

    def get_direct_access(self, remote_path, **_):
        return None

    def test_upload(self, test_path, config, **_):
        return True


class _GoogleCloudStorageDriver(_Driver):
    """Storage driver for google cloud storage"""

    _stream_download_pool_connections = 128
    _stream_download_pool = None

    _containers = {}

    scheme = 'gs'
    scheme_prefix = str(furl(scheme=scheme, netloc=''))

    class _Container(object):
        def __init__(self, name, cfg):
            try:
                from google.cloud import storage
                from google.oauth2 import service_account
            except ImportError:
                raise UsageError(
                    'Google cloud driver not found. '
                    'Please install driver using "pip install \'google-cloud-storage>=1.13.2\'"'
                )

            self.name = name[len(_GoogleCloudStorageDriver.scheme_prefix):]

            if cfg.credentials_json:
                credentials = service_account.Credentials.from_service_account_file(cfg.credentials_json)
            else:
                credentials = None

            self.client = storage.Client(project=cfg.project, credentials=credentials)
            self.config = cfg
            self.bucket = self.client.bucket(self.name)

    def _get_stream_download_pool(self):
        if self._stream_download_pool is None:
            self._stream_download_pool = ThreadPoolExecutor(max_workers=self._stream_download_pool_connections)
        return self._stream_download_pool

    def get_container(self, container_name, config=None, **kwargs):
        if container_name not in self._containers:
            self._containers[container_name] = self._Container(name=container_name, cfg=config)
        self._containers[container_name].config.retries = kwargs.get('retries', 5)
        return self._containers[container_name]

    def upload_object_via_stream(self, iterator, container, object_name, extra=None, **kwargs):
        try:
            blob = container.bucket.blob(object_name)
            blob.upload_from_file(iterator)
        except Exception as ex:
            log.error('Failed uploading: %s' % ex)
            return False
        return True

    def upload_object(self, file_path, container, object_name, extra=None, **kwargs):
        try:
            blob = container.bucket.blob(object_name)
            blob.upload_from_filename(file_path)
        except Exception as ex:
            log.error('Failed uploading: %s' % ex)
            return False
        return True

    def list_container_objects(self, container, **kwargs):
        return list(container.bucket.list_blobs())

    def delete_object(self, object, **kwargs):
        object.delete()

    def get_object(self, container_name, object_name, *args, **kwargs):
        full_container_name = str(furl(scheme=self.scheme, netloc=container_name))
        container = self._containers[full_container_name]
        obj = container.bucket.blob(object_name)
        obj.container_name = full_container_name
        return obj

    def download_object_as_stream(self, obj, chunk_size=256 * 1024, **_):
        raise NotImplementedError('Unsupported for google storage')

        def async_download(a_obj, a_stream):
            try:
                a_obj.download_to_file(a_stream)
            except Exception as ex:
                log.error('Failed downloading: %s' % ex)
            a_stream.close()

        # return iterable object
        stream = _Stream()
        obj.chunk_size = chunk_size
        self._get_stream_download_pool().submit(async_download, obj, stream)

        return stream

    def download_object(self, obj, local_path, overwrite_existing=True, delete_on_failure=True, callback=None, **_):
        p = Path(local_path)
        if not overwrite_existing and p.is_file():
            log.warning('failed saving after download: overwrite=False and file exists (%s)' % str(p))
            return
        obj.download_to_filename(str(p))

    def test_upload(self, test_path, config, **_):
        bucket_url = str(furl(scheme=self.scheme, netloc=config.bucket, path=config.subdir))
        bucket = self.get_container(container_name=bucket_url, config=config).bucket

        test_obj = bucket

        if test_path:
            if not test_path.endswith('/'):
                test_path += '/'

            blob = bucket.blob(test_path)

            if blob.exists():
                test_obj = blob

        permissions_to_test = ('storage.objects.get', 'storage.objects.update')
        return set(test_obj.test_iam_permissions(permissions_to_test)) == set(permissions_to_test)

    def get_direct_access(self, remote_path, **_):
        return None


class _AzureBlobServiceStorageDriver(_Driver):
    scheme = 'azure'

    _containers = {}

    class _Container(object):
        def __init__(self, name, config):
            try:
                from azure.common import AzureHttpError
                from azure.storage.blob import BlockBlobService
            except ImportError:
                raise UsageError(
                    'Azure blob storage driver not found. '
                    'Please install driver using "pip install \'azure.storage.blob>=2.0.1\'"'
                )

            self.name = name
            self.config = config
            self.blob_service = BlockBlobService(
                account_name=config.account_name,
                account_key=config.account_key,
            )

    @attrs
    class _Object(object):
        container = attrib()
        blob_name = attrib()
        content_length = attrib()

    def get_container(self, container_name=None, config=None, **kwargs):
        container_name = container_name or config.container_name
        if container_name not in self._containers:
            self._containers[container_name] = self._Container(name=container_name, config=config)
        # self._containers[container_name].config.retries = kwargs.get('retries', 5)
        return self._containers[container_name]

    def upload_object_via_stream(self, iterator, container, object_name, extra=None, **kwargs):
        from azure.common import AzureHttpError

        blob_name = self._blob_name_from_object_path(object_name, container.name)
        try:
            container.blob_service.MAX_SINGLE_PUT_SIZE = 16 * 1024 * 1024
            container.blob_service.socket_timeout = (300, 2000)
            container.blob_service.create_blob_from_bytes(
                container.name,
                object_name,
                bytes(iterator),
                # timeout=300,
                max_connections=2,
            )
            return True
        except AzureHttpError as ex:
            log.error('Failed uploading (Azure error): %s' % ex)
        except Exception as ex:
            log.error('Failed uploading: %s' % ex)
        return False

    def upload_object(self, file_path, container, object_name, extra=None, **kwargs):
        from azure.common import AzureHttpError

        blob_name = self._blob_name_from_object_path(object_name, container.name)
        stream = None
        try:
            from azure.storage.blob import ContentSettings
            from mimetypes import guess_type
            container.blob_service.MAX_SINGLE_PUT_SIZE = 16 * 1024 * 1024
            container.blob_service.socket_timeout = (300, 2000)
            container.blob_service.create_blob_from_path(
                container.name,
                blob_name,
                file_path,
                # timeout=300,
                max_connections=2,
                content_settings=ContentSettings(content_type=guess_type(file_path))
            )
            return True
        except AzureHttpError as ex:
            log.error('Failed uploading (Azure error): %s' % ex)
        except Exception as ex:
            log.error('Failed uploading: %s' % ex)
        finally:
            if stream:
                stream.close()

    def list_container_objects(self, container, ex_prefix=None, **kwargs):
        return list(container.blob_service.list_blobs(container_name=container.name, prefix=ex_prefix))

    def delete_object(self, object, **kwargs):
        container = object.container
        container.blob_service.delete_blob(
            container.name,
            object.blob_name,
        )

    def get_object(self, container_name, object_name, *args, **kwargs):
        container = self._containers.get(container_name)
        if not container:
            raise StorageError("Container `{}` not found for object {}".format(container_name, object_name))

        # blob_name = self._blob_name_from_object_path(object_name, container_name)
        blob = container.blob_service.get_blob_properties(container.name, object_name)

        return self._Object(container=container, blob_name=blob.name, content_length=blob.properties.content_length)

    def download_object_as_stream(self, obj, *_, **__):
        container = obj.container
        blob = container.blob_service.get_blob_to_bytes(
            container.name,
            container.blob_name,
        )
        return blob.content

    def download_object(self, obj, local_path, overwrite_existing=True, delete_on_failure=True, callback=None, **_):
        p = Path(local_path)
        if not overwrite_existing and p.is_file():
            log.warning('failed saving after download: overwrite=False and file exists (%s)' % str(p))
            return

        download_done = threading.Event()
        download_done.counter = 0

        def callback_func(current, total):
            if callback:
                chunk = current-download_done.counter
                download_done.counter += chunk
                callback(chunk)
            if current >= total:
                download_done.set()

        container = obj.container
        container.blob_service.MAX_SINGLE_GET_SIZE = 5 * 1024 * 1024
        _ = container.blob_service.get_blob_to_path(
            container.name,
            obj.blob_name,
            local_path,
            max_connections=10,
            progress_callback=callback_func,
        )
        download_done.wait()

    def test_upload(self, test_path, config, **_):
        container = self.get_container(config=config)
        try:
            container.blob_service.get_container_properties(container.name)
        except Exception:
            return False
        else:
            # Using the account Key, we can always upload...
            return True

    @classmethod
    def _blob_name_from_object_path(cls, name, container_name):
        scheme = urlparse(name).scheme
        if scheme:
            if scheme != cls.scheme:
                raise StorageError(
                    "When using a URL, only the `{}` scheme is supported for Azure storage: {}",
                    cls.scheme,
                    name,
                )

            f = furl(name)

            if not f.path.segments:
                raise StorageError(
                    "Missing container name in URL {}",
                    name,
                )

            parsed_container_name = f.path.segments[0]

            if parsed_container_name != container_name:
                raise StorageError(
                    "Container name mismatch (expected {}, found {}) in {}",
                    container_name,
                    parsed_container_name,
                    name,
                )

            if len(f.path.segments) == 1:
                raise StorageError(
                    "No path found following container name {} in {}",
                    container_name,
                    name,
                )

            return f.path.segments[0], os.path.join(*f.path.segments[1:])

        return name

    def get_direct_access(self, remote_path, **_):
        return None


class _FileStorageDriver(_Driver):
    """
    A base StorageDriver to derive from.
    """

    scheme = "file"
    CHUNK_SIZE = 8096
    IGNORE_FOLDERS = ['.lock', '.hash']
    Object = namedtuple("Object", ['name', 'size', 'extra', 'driver', 'container', 'hash', 'meta_data'])

    class _Container(object):
        def __init__(self, name, extra, driver):
            self.name = name
            self.extra = extra
            self.driver = driver

    def __init__(self, key, secret=None, secure=True, host=None, port=None,
                 **kwargs):

        # Use the key as the path to the storage
        self.base_path = key

        if not os.path.isdir(self.base_path):
            raise ValueError('The base path is not a directory')

    def _make_path(self, path, ignore_existing=True):
        """
        Create a path by checking if it already exists
        """

        try:
            os.makedirs(path)
        except OSError:
            exp = sys.exc_info()[1]
            if exp.errno == errno.EEXIST and not ignore_existing:
                raise exp

    def _check_container_name(self, container_name):
        """
        Check if the container name is valid

        :param container_name: Container name
        :type container_name: ``str``
        """

        if '/' in container_name or '\\' in container_name:
            raise ValueError(value=None, driver=self, container_name=container_name)

    def _make_container(self, container_name):
        """
        Create a container instance

        :param container_name: Container name.
        :type container_name: ``str``

        :return: Container instance.
        :rtype: :class:`Container`
        """

        self._check_container_name(container_name)

        full_path = os.path.realpath(os.path.join(self.base_path, container_name))

        try:
            stat = os.stat(full_path)
            if not os.path.isdir(full_path):
                raise OSError('Target path is not a directory')
        except OSError:
            raise ValueError(value=None, driver=self, container_name=container_name)

        extra = {}
        extra['creation_time'] = stat.st_ctime
        extra['access_time'] = stat.st_atime
        extra['modify_time'] = stat.st_mtime

        return self._Container(name=container_name, extra=extra, driver=self)

    def _make_object(self, container, object_name):
        """
        Create an object instance

        :param container: Container.
        :type container: :class:`Container`

        :param object_name: Object name.
        :type object_name: ``str``

        :return: Object instance.
        :rtype: :class:`Object`
        """

        full_path = os.path.realpath(os.path.join(self.base_path, container.name, object_name))

        if os.path.isdir(full_path):
            raise ValueError(value=None, driver=self, object_name=object_name)

        try:
            stat = os.stat(full_path)
        except Exception:
            raise ValueError(value=None, driver=self, object_name=object_name)

        extra = {}
        extra['creation_time'] = stat.st_ctime
        extra['access_time'] = stat.st_atime
        extra['modify_time'] = stat.st_mtime

        return self.Object(name=object_name, size=stat.st_size, extra=extra,
                           driver=self, container=container, hash=None, meta_data=None)

    def iterate_containers(self):
        """
        Return a generator of containers.

        :return: A generator of Container instances.
        :rtype: ``generator`` of :class:`Container`
        """

        for container_name in os.listdir(self.base_path):
            full_path = os.path.join(self.base_path, container_name)
            if not os.path.isdir(full_path):
                continue
            yield self._make_container(container_name)

    def _get_objects(self, container):
        """
        Recursively iterate through the file-system and return the object names
        """

        cpath = self.get_container_cdn_url(container, check=True)

        for folder, subfolders, files in os.walk(cpath, topdown=True):
            # Remove unwanted subfolders
            for subf in self.IGNORE_FOLDERS:
                if subf in subfolders:
                    subfolders.remove(subf)

            for name in files:
                full_path = os.path.join(folder, name)
                object_name = os.path.relpath(full_path, start=cpath)
                yield self._make_object(container, object_name)

    def iterate_container_objects(self, container):
        """
        Returns a generator of objects for the given container.

        :param container: Container instance
        :type container: :class:`Container`

        :return: A generator of Object instances.
        :rtype: ``generator`` of :class:`Object`
        """

        return self._get_objects(container)

    def get_container(self, container_name, **_):
        """
        Return a container instance.

        :param container_name: Container name.
        :type container_name: ``str``

        :return: :class:`Container` instance.
        :rtype: :class:`Container`
        """
        return self._make_container(container_name)

    def get_container_cdn_url(self, container, check=False):
        """
        Return a container CDN URL.

        :param container: Container instance
        :type  container: :class:`Container`

        :param check: Indicates if the path's existence must be checked
        :type check: ``bool``

        :return: A CDN URL for this container.
        :rtype: ``str``
        """
        path = os.path.realpath(os.path.join(self.base_path, container.name))

        if check and not os.path.isdir(path):
            raise ValueError(value=None, driver=self, container_name=container.name)

        return path

    def get_object(self, container_name, object_name, **_):
        """
        Return an object instance.

        :param container_name: Container name.
        :type  container_name: ``str``

        :param object_name: Object name.
        :type  object_name: ``str``

        :return: :class:`Object` instance.
        :rtype: :class:`Object`
        """
        container = self._make_container(container_name)
        return self._make_object(container, object_name)

    def get_object_cdn_url(self, obj):
        """
        Return an object CDN URL.

        :param obj: Object instance
        :type  obj: :class:`Object`

        :return: A CDN URL for this object.
        :rtype: ``str``
        """
        return os.path.realpath(os.path.join(self.base_path, obj.container.name, obj.name))

    def download_object(self, obj, destination_path, overwrite_existing=False, delete_on_failure=True, **_):
        """
        Download an object to the specified destination path.

        :param obj: Object instance.
        :type obj: :class:`Object`

        :param destination_path: Full path to a file or a directory where the
                                incoming file will be saved.
        :type destination_path: ``str``

        :param overwrite_existing: True to overwrite an existing file,
            defaults to False.
        :type overwrite_existing: ``bool``

        :param delete_on_failure: True to delete a partially downloaded file if
        the download was not successful (hash mismatch / file size).
        :type delete_on_failure: ``bool``

        :return: True if an object has been successfully downloaded, False
        otherwise.
        :rtype: ``bool``
        """

        obj_path = self.get_object_cdn_url(obj)
        base_name = os.path.basename(destination_path)

        if not base_name and not os.path.exists(destination_path):
            raise ValueError(
                value='Path %s does not exist' % (destination_path),
                driver=self)

        if not base_name:
            file_path = os.path.join(destination_path, obj.name)
        else:
            file_path = destination_path

        if os.path.exists(file_path) and not overwrite_existing:
            raise ValueError('File %s already exists, but ' % (file_path) + 'overwrite_existing=False')

        try:
            shutil.copy(obj_path, file_path)
        except IOError:
            if delete_on_failure:
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
            return False

        return True

    def download_object_as_stream(self, obj, chunk_size=None, **_):
        """
        Return a generator which yields object data.

        :param obj: Object instance
        :type obj: :class:`Object`

        :param chunk_size: Optional chunk size (in bytes).
        :type chunk_size: ``int``

        :return: A stream of binary chunks of data.
        :rtype: ``object``
        """
        path = self.get_object_cdn_url(obj)
        with open(path, 'rb') as obj_file:
            for data in self._read_in_chunks(obj_file, chunk_size=chunk_size):
                yield data

    def upload_object(self, file_path, container, object_name, extra=None, verify_hash=True, **_):
        """
        Upload an object currently located on a disk.

        :param file_path: Path to the object on disk.
        :type file_path: ``str``

        :param container: Destination container.
        :type container: :class:`Container`

        :param object_name: Object name.
        :type object_name: ``str``

        :param verify_hash: Verify hast
        :type verify_hash: ``bool``

        :param extra: (optional) Extra attributes (driver specific).
        :type extra: ``dict``

        :rtype: ``object``
        """

        path = self.get_container_cdn_url(container, check=True)
        obj_path = os.path.join(path, object_name)
        base_path = os.path.dirname(obj_path)

        self._make_path(base_path)

        shutil.copy(file_path, obj_path)

        os.chmod(obj_path, int('664', 8))

        return self._make_object(container, object_name)

    def upload_object_via_stream(self, iterator, container, object_name, extra=None, **kwargs):
        """
        Upload an object using an iterator.

        If a provider supports it, chunked transfer encoding is used and you
        don't need to know in advance the amount of data to be uploaded.

        Otherwise if a provider doesn't support it, iterator will be exhausted
        so a total size for data to be uploaded can be determined.

        Note: Exhausting the iterator means that the whole data must be
        buffered in memory which might result in memory exhausting when
        uploading a very large object.

        If a file is located on a disk you are advised to use upload_object
        function which uses fs.stat function to determine the file size and it
        doesn't need to buffer whole object in the memory.

        :type iterator: ``object``
        :param iterator: An object which implements the iterator
                         interface and yields binary chunks of data.

        :type container: :class:`Container`
        :param container: Destination container.

        :type object_name: ``str``
        :param object_name: Object name.

        :type extra: ``dict``
        :param extra: (optional) Extra attributes (driver specific). Note:
            This dictionary must contain a 'content_type' key which represents
            a content type of the stored object.

        :rtype: ``object``
        """
        path = self.get_container_cdn_url(container, check=True)
        obj_path = os.path.join(path, object_name)
        base_path = os.path.dirname(obj_path)
        self._make_path(base_path)

        obj_path = os.path.realpath(obj_path)
        with open(obj_path, 'wb') as obj_file:
            obj_file.write(iterator.read() if hasattr(iterator, 'read') else bytes(iterator))

        os.chmod(obj_path, int('664', 8))
        return self._make_object(container, object_name)

    def delete_object(self, obj, **_):
        """
        Delete an object.

        :type obj: :class:`Object`
        :param obj: Object instance.

        :return: ``bool`` True on success.
        :rtype: ``bool``
        """

        path = self.get_object_cdn_url(obj)

        try:
            os.unlink(path)
        except Exception:
            return False

        # # Check and delete all the empty parent folders
        # path = os.path.dirname(path)
        # container_url = obj.container.get_cdn_url()
        #
        # # Delete the empty parent folders till the container's level
        # while path != container_url:
        #     try:
        #         os.rmdir(path)
        #     except OSError:
        #         exp = sys.exc_info()[1]
        #         if exp.errno == errno.ENOTEMPTY:
        #             break
        #         raise exp
        #
        #     path = os.path.dirname(path)

        return True

    def create_container(self, container_name):
        """
        Create a new container.

        :type container_name: ``str``
        :param container_name: Container name.

        :return: :class:`Container` instance on success.
        :rtype: :class:`Container`
        """

        self._check_container_name(container_name)

        path = os.path.join(self.base_path, container_name)

        try:
            self._make_path(path, ignore_existing=False)
        except OSError:
            exp = sys.exc_info()[1]
            if exp.errno == errno.EEXIST:
                raise ValueError('Container %s with this name already exists. The name '
                                 'must be unique among all the containers in the '
                                 'system' % container_name)
            else:
                raise ValueError( 'Error creating container %s' % container_name)
        except Exception:
            raise ValueError('Error creating container %s' % container_name)

        return self._make_container(container_name)

    def delete_container(self, container):
        """
        Delete a container.

        :type container: :class:`Container`
        :param container: Container instance

        :return: True on success, False otherwise.
        :rtype: ``bool``
        """

        # Check if there are any objects inside this
        for obj in self._get_objects(container):
            raise ValueError(value='Container %s is not empty' % container.name)

        path = self.get_container_cdn_url(container, check=True)

        try:
            shutil.rmtree(path)
        except Exception:
            return False

        return True

    def list_container_objects(self, container, **kwargs):
        return list(self.iterate_container_objects(container))

    @staticmethod
    def _read_in_chunks(iterator, chunk_size=None, fill_size=False, yield_empty=False):
        """
        Return a generator which yields data in chunks.

        :param iterator: An object which implements an iterator interface
                         or a File like object with read method.
        :type iterator: :class:`object` which implements iterator interface.

        :param chunk_size: Optional chunk size (defaults to CHUNK_SIZE)
        :type chunk_size: ``int``

        :param fill_size: If True, make sure chunks are exactly chunk_size in
                          length (except for last chunk).
        :type fill_size: ``bool``

        :param yield_empty: If true and iterator returned no data, only yield empty
                            bytes object
        :type yield_empty: ``bool``

        TODO: At some point in the future we could use byte arrays here if version
        >= Python 3. This should speed things up a bit and reduce memory usage.
        """
        chunk_size = chunk_size or _FileStorageDriver.CHUNK_SIZE
        if six.PY3:
            from io import FileIO as file

        if isinstance(iterator, (file)):
            get_data = iterator.read
            args = (chunk_size,)
        else:
            get_data = next
            args = (iterator,)

        data = bytes('')
        empty = False

        while not empty or len(data) > 0:
            if not empty:
                try:
                    chunk = bytes(get_data(*args))
                    if len(chunk) > 0:
                        data += chunk
                    else:
                        empty = True
                except StopIteration:
                    empty = True

            if len(data) == 0:
                if empty and yield_empty:
                    yield bytes('')

                return

            if fill_size:
                if empty or len(data) >= chunk_size:
                    yield data[:chunk_size]
                    data = data[chunk_size:]
            else:
                yield data
                data = bytes('')

    def get_direct_access(self, remote_path, **_):
        # this will always make sure we have full path and file:// prefix
        full_url = StorageHelper.conform_url(remote_path)
        # now get rid of the file:// prefix
        return Path(full_url[7:]).as_posix()

    def test_upload(self, test_path, config, **kwargs):
        return True


driver_schemes = set(
    filter(
        None,
        itertools.chain(
            (getattr(cls, "scheme", None) for cls in _Driver.__subclasses__()),
            *(getattr(cls, "schemes", []) for cls in _Driver.__subclasses__())
        )
    )
)

remote_driver_schemes = driver_schemes - {_FileStorageDriver.scheme}
