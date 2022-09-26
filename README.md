## Description
* Add PYPI_USERNAME and PYPI_PASSWORD to your secrets using GitHub GUI. This is required to push your package to PyPI.
* Run `customize.py` to take care of the rest.
* Each time you push a new release, code is automatically published on PyPI via the workflow. `pip install -U <your_package>` should then install the latest version of your package.
