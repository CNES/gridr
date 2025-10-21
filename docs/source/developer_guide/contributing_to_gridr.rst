
.. _contributing_to_gridr:

=====================
Contributing to GRIDR
=====================

GRIDR is an open source software : don't hesitate to hack it and contribute !

Please go to `the GitHub repository <https://www.github.com/CNES/gridr>`_  for source code.

Read also `GRIDR Contribution guide <https://www.github.com/CNES/gridr/tree/main/CONTRIBUTING.md>`_ with `LICENCE <https://raw.githubusercontent.com/CNES/gridr/main/LICENSE>`_ and `Contributor Licence Agrements <https://github.com/CNES/gridr/tree/main/docs/source/CLA>`_.

**Contact:** |contact_email| 

Developer Install
=================

We recommend to use a `virtualenv` environment, so that `GRIDR` do not interfere with other packages installed on your system.

* Clone GRIDR repository from GitHub :

.. code-block:: console

    git clone https://github.com/CNES/gridr.git
    cd gridr
    make venv


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


Branching Strategy : a Strict Linear Variant of Trunk-Based Development
=======================================================================

Overview
--------
This section outlines our **Linear Branch Model**, a Git workflow designed to maintain a **clean, linear history** in the ``main`` branch. Unlike traditional branching models (e.g., GitFlow), this approach enforces strict rebase and squash rules to ensure:

- **No merge commits** in ``main``
- **No direct pushes** to ``main``
- **All changes** come from feature branches, rebased and squashed before merging

This model is inspired by **Trunk-Based Development** but with stricter rebase and squash requirements.

Main Branch: ``main``
---------------------
**Purpose**
  - The **only protected branch** in the repository
  - Represents the **stable, production-ready** state of the codebase
  - **Must always be linear** (no merge commits)

**Rules**

1. **No direct pushes**

   - All changes must come from feature branches
   - Use **fast-forward merges** where possible

2. **Rebase before merging**

   - Feature branches must be rebased onto ``main`` before merging
   - Use ``git rebase -i`` to squash unnecessary commits

3. **Fast-forward merges only**

   - If a fast-forward merge is not possible, the branch must be rebased further to resolve conflicts

Feature Branches
----------------
**Purpose**
  - Used for **all new development**
  - Must be **rebased and squashed** before merging into ``main``

**Rules**

1. **Create from ``main``**

   .. code-block:: bash

      git checkout main
      git fetch origin
      git rebase origin/main
      git checkout -b feature/your-feature

2. **Rebase before merging**

   - Fetch latest changes from remote:

   .. code-block:: bash

      git fetch origin

   - Rebase onto ``main`` to incorporate the latest changes:

   .. code-block:: bash

      git checkout feature/your-feature
      git rebase origin/main

   - Squash unnecessary commits with ``git rebase -i HEAD~N`` (replace ``N`` with the number of commits to squash)

3. **No ``WIP`` commits**

   - All commits must be **meaningful and final**
   - Use ``git commit --amend`` to fix commit messages

4. **Force push after rebase**

   - After rebasing, force push to update the remote branch:

   .. code-block:: bash

      git push origin feature/your-feature --force-with-lease

Workflow Steps
--------------

1. **Create a Feature Branch**

.. code-block:: bash

   git checkout main
   git fetch origin
   git rebase origin/main
   git checkout -b xx-your-feature  # Prefix with issue number if related to an issue

Open a Merge Request as soon as possible with:
   - Title: ``WIP: [Feature] xx-your-feature`` (if work in progress)
   - Description: Short description of changes + ``Closes #xx`` if related to an issue

2. **Develop Your Feature**

   - Make changes and commit them with clear, descriptive messages
   - Avoid ``WIP`` or temporary commits
   - Each commit should represent a complete, testable change

3. **Prepare for Merge**

   a. **Fetch latest changes**:

      .. code-block:: bash

         git fetch origin

   b. **Rebase onto main**:

      .. code-block:: bash

         git checkout xx-your-feature
         git rebase origin/main

   c. **Squash commits** (if needed):

      .. code-block:: bash

         git rebase -i HEAD~N  # Replace N with number of commits to squash

      In the interactive editor:
        - Mark commits with ``squash``
        - Example:

      .. code-block:: bash

         pick abc123 First commit
         squash def456 Second commit

4. **Force Push**

.. code-block:: bash

   git push origin xx-your-feature --force-with-lease

5. **Finalize Merge via GitHub**

   - After preparing your branch (rebased and squashed):
   - Go to GitHub and create/access the Pull Request
   - Ensure all checks pass.
   - Remove ``WIP:`` prefix from the title if present
   - Request review from at least one maintainer
   - Wait for approval from a maintainer
   - Once approved, click "Merge pull request" button
   - Delete the feature branch after successful merge

   .. note::
      - The actual merge operation requires maintainer approval
      - The commands below are for local cleanup after merge
      - Maintainers can merge directly after approval

.. code-block:: bash

   git fetch origin
   git checkout main
   git rebase origin/main
   git branch -d xx-your-feature  # Delete local branch

Important Rules
---------------

1. **Protected Main Branch**

   - Any code modification requires a Merge Request
   - It is forbidden to push patches directly into main
   - This branch is protected against direct pushes

2. **Merge Request Best Practices**

   - Open Merge Requests as soon as possible to inform developers
   - Use ``WIP:`` prefix for work in progress to prevent accidental merges
   - Provide a short description of proposed changes
   - Link to issues using ``Closes #xx``
   - Prefix branch names with issue numbers (``xx-``)

3. **Commit Quality**

   - No temporary or WIP commits in final branch
   - Each commit must be meaningful and testable
   - Use ``git commit --amend`` to fix commit messages
   - Squash unnecessary commits before merging
   

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
