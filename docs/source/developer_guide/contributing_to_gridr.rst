
.. _contributing_to_cars:

=====================
Contributing to GRIDR
=====================

GRIDR is an open source software : don't hesitate to hack it and contribute !

Please go to `the GitHub repository <https://www.github.com/CNES/gridr>`_  for source code.

Read also `GRIDR Contribution guide <https://www.github.com/CNES/gridr/tree/master/CONTRIBUTING.md>`_ with `LICENCE <https://raw.githubusercontent.com/CNES/gridr/master/LICENSE>`_ and `Contributor Licence Agrements <https://github.com/CNES/gridr/tree/master/docs/source/CLA>`_.

**Contact:** |contact_email| 

Developer Install
=================

We recommend to use a `virtualenv` environment, so that `GRIDR` do not interfere with other packages installed on your system.

* Clone GRIDR repository from GitHub :

.. code-block:: console

    git clone https://github.com/CNES/gridr.git
    cd gridr 

(TODO) details


Coding guide
============

Here are some rules to apply when developing a new functionality:

* **Comments:** Include a comments ratio high enough and use explicit variables names. A comment by code block of several lines is necessary to explain a new functionality.
* **Test**: Each new functionality shall have a corresponding test in its module's test file. This test shall, if possible, check the function's outputs and the corresponding degraded cases.
* **Documentation**: All functions shall be documented (object, parameters, return values).
* **Use type hints**: Use the type hints provided by the `typing` python module.
* **Use doctype**: Follow sphinx default doctype for automatic API.
* **Quality code**: Correct project quality code errors with pre-commit automatic workflow (see below).
* **Factorization**: Factorize the code as much as possible. The command line tools shall only include the main workflow and rely on the cars python modules.
* **Be careful with user interface upgrade:** If major modifications of the user interface or of the tool's behaviour are done, update the user documentation (and the notebooks if necessary).
* **Logging and no print**: The usage of the `print()` function is forbidden: use the `logging` python standard module instead.
* **Limit classes**: If possible, limit the use of classes at one or 2 levels and opt for a functional approach when possible. The classes are reserved for data modelling if it is impossible to do so using `xarray` and for the good level of modularity.
* **Limit new dependencies**: Do not add new dependencies unless it is absolutely necessary, and only if it has a **permissive license**.

Pre-commit validation
=====================

A pre-commit validation is installed with code quality tools (see below).
It is installed automatically by `make venv` command.

Here is the way to install it manually:

.. code-block:: console

  pre-commit install -t pre-commit # for commit rules
  pre-commit install -t pre-push   # for push rules

This installs the pre-commit hook in `.git/hooks/pre-commit` and `.git/hooks/pre-push`  from `.pre-commit-config.yaml <https://raw.githubusercontent.com/CNES/gridr/master/.pre-commit-config.yaml>`_ file configuration.

It is possible to test pre-commit before committing:

.. code-block:: console

  pre-commit run --all-files                        # Run all hooks on all files
  pre-commit run --files python/gridr/__init__.py   # Run all hooks on one file
  pre-commit run pylint                             # Run only pylint hook


Gitflow
=======

Main branches
-------------

The development model is greatly inspired by existing models such as `nvie's model <https://nvie.com/posts/a-successful-git-branching-model/>`_. The central repository holds two main branches with an infinite lifetime:

- ``main``
- ``dev``

The stable production branch: ``origin/main``
---------------------------------------------

Branch ``origin/main`` is considered here as the default branch. The source code of ``HEAD`` should reflect a stable production state. Therefore some rules must be respected:

- Do not directly commit to it.
- Update to the ``origin/main`` branch should only be made through a merge request from the ``origin/dev`` branch (or an optional release branch)

To summarize: your ``origin/main`` branch is your "do not touch it clean and stable branch".

The developers integration branch: ``origin/dev``
-------------------------------------------------

Branch ``origin/dev`` is considered to hold a source code of ``HEAD`` that is always reflecting a state with the latest delivered and integrated changes for the next release: it is the ``integration branch`` on which developers are merging to.

Here are some rules to respect:

- Do not directly push to this branch but perform merge request from "a new feature branch" to "origin/dev".
- After the ``origin/dev`` branch has been merged into ``origin/main``, the ``origin/dev`` must be updated in order to be in the same state as the ``master``. This is performed in your local development repo:

.. code-block:: bash

   git checkout dev
   git fetch origin
   git merge origin/main

When the source code in the ``dev`` branch reaches a stable point and is ready to be released, all of the changes should be merged back into ``main`` through a ``merge request``. When you are ready for a new release you should perform a merge request from 'dev' to 'main'.

Supporting branches
-------------------

Next to the main branches ``main`` and ``develop``, our development model uses a variety of supporting branches to aid parallel development between team members, ease tracking of features, prepare for production releases and to assist in quickly fixing live production problems.

Unlike the main branches, these branches always have a limited life time, since they will be removed eventually. The different types of branches we may use are:

- Feature branches
- Release branches
- Hotfix branches

Each of these branches have a specific purpose and are bound to strict rules as to which branches may be their originating branch and which branches must be their merge targets. By no means are these branches "special" from a technical perspective. The branch types are categorized by how we use them. They are of course plain old Git branches.

In order to make it simple we will only force you to use ``feature`` branches. Releases and associated tags are automatically created through CI.

New features branch
-------------------

Create a new feature specific branch from the 'dev' branch for each new feature.

It is recommended to open your Merge Request as soon as possible in order to inform the developers of your ongoing work.
Please add `WIP:` before your Merge Request title if your work is in progress: This prevents an accidental merge and informs the other developers of the unfinished state of your work.

The Merge Request shall have a short description of the proposed changes. If it is relative to an issue, you can signal it by adding `Closes xx` where xx is the reference number of the is
sue.

Prefix the branch's name by `xx-` in order to link it to the xx issue.


Documentation
=============

GRIDR contains its Sphinx Documentation in the code in docs directory.

To generate documentation, use:

.. code-block:: console

  make build-sphinx-doc

The documentation is then build in docs/build directory and can be consulted with a web browser.

Documentation can be edited in docs/source/ RST files.

Jupyter notebooks
=================

GRIDR contains notebooks in tutorials directory.


Code quality
=============
GRIDR uses `Isort`_, `Black`_, `Flake8`_ and `Pylint`_ quality code checking.


Isort
-----
`Isort`_ is a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type.

GRIDR ``isort`` configuration is done in `pyproject.toml`

`Isort`_ manual usage examples:

.. code-block:: console

    isort --check python/gridr tests/python  # Check code with isort, does nothing
    isort --diff python/gridr tests/python   # Show isort diff modifications
    isort python/gridr tests/python          # Apply modifications

`Isort`_ messages can be avoided when really needed with **"# isort:skip"** on the incriminated line.

Black
-----
`Black`_ is a quick and deterministic code formatter to help focus on the content.

GRIDR ``black`` configuration is done in `pyproject.toml`

If necessary, Black doesn’t reformat blocks that start with "# fmt: off" and end with # fmt: on, or lines that ends with "# fmt: skip". "# fmt: on/off" have to be on the same level of indentation.

`Black`_ manual usage examples:

.. code-block:: console

    black --check python/gridr tests/python  # Check code with black with no modifications
    black --diff python/gridr tests/python   # Show black diff modifications
    black python/gridr tests/python          # Apply modifications

Flake8
------
`Flake8`_ is a command-line utility for enforcing style consistency across Python projects. By default it includes lint checks provided by the `PyFlakes project <https://github.com/PyCQA/pyflakes>`_ , PEP-0008 inspired style checks provided by the `PyCodeStyle project <https://github.com/PyCQA/pycodestyle>`_ , and McCabe complexity checking provided by the `McCabe project <https://github.com/PyCQA/mccabe>`_. It will also run third-party extensions if they are found and installed.

GRIDR ``flake8`` configuration is done in `pyproject.toml`

`Flake8`_ messages can be avoided (in particular cases !) adding "# noqa" in the file or line for all messages.
It is better to choose filter message with "# noqa: E731" (with E371 example being the error number).
Look at examples in source code.

Flake8 manual usage examples:

.. code-block:: console

  flake8 python/gridr tests/python           # Run all flake8 tests


Pylint
------
`Pylint`_ is a global linting tool which helps to have many information on source code.

GRIDR ``pylint`` configuration is done in dedicated `.pylintrc <//https://raw.githubusercontent.com/CNES/gridr/master/.pylintrc_RNC2015_D>`_ file.

`Pylint`_ messages can be avoided (in particular cases !) adding "# pylint: disable=error-message-name" in the file or line.
Look at examples in source code.

Pylint manual usage examples:

.. code-block:: console

  pylint tests/python python/gridr       # Run all pylint tests
  pylint --list-msgs                     # Get pylint detailed errors information


Tests
======

GRIDR includes a set of tests executed with `pytest <https://docs.pytest.org/>`_ tool.

To launch tests:

.. code-block:: console

    make test
