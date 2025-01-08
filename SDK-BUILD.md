# Steps
1. vi ~/.pypirc # configure [nexus] to the internal pypi repo creds
```[nexus]
repository = https://path/to/pypi-hosted-repo
username = deploymentUserName
password = deploymentPassowrd
```
2. python -m pip install build
3. python -m pip install twine
4. Update version in clearml/version.py
5. python -m build --wheel
6. twine upload --repository nexus dist/*
7. python setup.py clean