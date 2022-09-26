import os

# Manually modify following parameters to customize the structure of your project
path = os.path.abspath(os.path.dirname(__file__)).split("/")
# print(path)
REPO_HOME_PATH = "/".join(path[:-1])
REPO_NAME = path[-1]
PACKAGE_NAME = REPO_NAME
AUTHOR = "Zeel B Patel"
AUTHOR_EMAIL = "patel_zeel@iitgn.ac.in"
description = "example description"
URL = "https://github.com/patel-zeel/" + REPO_NAME
LICENSE = "MIT"
LICENSE_FILE = "LICENSE"
LONG_DESCRIPTION = "file: README.md"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

full_path = os.path.join(REPO_HOME_PATH, REPO_NAME)

# Write setup.cfg

with open(os.path.join(full_path, "setup.cfg"), "w") as f:
    f.write("[metadata]\n")
    f.write("name = " + PACKAGE_NAME + "\n")
    f.write("author = " + AUTHOR + "\n")
    f.write("author-email = " + AUTHOR_EMAIL + "\n")
    f.write("description = " + description + "\n")
    f.write("url = " + URL + "\n")
    f.write("license = " + LICENSE + "\n")
    f.write("long_description_content_type = " + LONG_DESCRIPTION_CONTENT_TYPE + "\n")
    f.write("long_description = " + LONG_DESCRIPTION + "\n")

# Write CI

with open(os.path.join(full_path, ".github/workflows/CI.template"), "r") as f:
    content = f.read()

with open(os.path.join(full_path, ".github/workflows/CI.yml"), "w") as f:
    content = content.replace("<reponame>", REPO_NAME)
    f.write(content)

# Write .gitignore
with open(os.path.join(full_path, ".gitignore"), "w") as f:
    f.write("__pycache__/\n")
    f.write("*.vscode\n")
    f.write("*.ipynb_checkpoints\n")
    f.write("*.pyc\n")
    f.write("*.egg-info/\n")
    f.write(f"{PACKAGE_NAME}/_version.py\n")


# Write pyproject.toml
with open(os.path.join(full_path, "pyproject.toml"), "w") as f:
    f.write("[build-system]\n")
    f.write("requires = [\n")
    f.write('\t"setuptools>=50.0",\n')
    f.write('\t"setuptools_scm[toml]>=6.0",\n')
    f.write('\t"setuptools_scm_git_archive",\n')
    f.write('\t"wheel>=0.33",\n')
    f.write('\t"numpy>=1.16",\n')
    f.write('\t"cython>=0.29",\n')
    f.write("\t]\n")
    f.write("\n")
    f.write("[tool.setuptools_scm]\n")
    f.write(f'write_to = "{PACKAGE_NAME}/_version.py"')

# Initialize project folder
os.makedirs(os.path.join(full_path , PACKAGE_NAME), exist_ok=True)
with open(os.path.join(full_path, PACKAGE_NAME, "__init__.py"), "w") as f:
    f.write("from ._version import version as __version__  # noqa")

print("Successful")
