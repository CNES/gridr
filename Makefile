
ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Project path to the python directory
GRIDR_PYTHON_PATH := $(ROOT_DIR)python/gridr
# Project path to the rust directory
GRIDR_RUST_CRATE_PATH := $(ROOT_DIR)rust/gridr

# In order to perform tests with pytest we have to build the rust library and to create
# a symbolic link to the target path
GRIDR_LIBGRIDR_SO_BUILD_TARGET := $(GRIDR_RUST_CRATE_PATH)/target/release/lib_libgridr.so
GRIDR_LIBGRIDR_SO_PYTEST_TARGET := $(GRIDR_PYTHON_PATH)/cdylib/_libgridr.so

GRIDR_LIBGRIDR_DOC_BUILD_PATH := $(GRIDR_RUST_CRATE_PATH)/target/doc

GRIDR_DOCS_ROOT_PATH := $(ROOT_DIR)docs

# Check cargo
CHECK_CARGO = $(shell command -v cargo 2> /dev/null)


# Check python3
PYTHON=$(shell command -v python3)
ifeq (, $(PYTHON))
	$(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif

# Check Python version
PYTHON_VERSION_MIN = 3.8
PYTHON_VERSION_CUR = $(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK = $(shell $(PYTHON) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')

#ifeq ($(PYTHON_VERSION_OK), 0)
#	$(error "Requires python version >= $(PYTHON_VERSION_MIN). Current version is $(PYTHON_VERSION_CUR)")
#endif

# Set Virtualenv directory name
# Default if not defined is to create venv with a name that contains the current python version
# Example: GRIDR_VENV="venv/other-venv/" make install
ifndef GRIDR_VENV
	GRIDR_VENV = "$(ROOT_DIR)venv/venv_py$(PYTHON_VERSION_CUR)"
endif

# PIP_ARG_MAIN is a variable that may contains PIP configuration arguments
# This variable may be set by the CI
ifndef PIP_ARG_MAIN
	PIP_ARG_MAIN := ""
endif

################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@echo "      GRIDR MAKE HELP"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort

.PHONY: check
check: ## check if cargo is installed
	@[ "$(CHECK_CARGO)" ] || ( echo ">> cargo not found"; exit 1 )

# Set virtual environment directory name
# You can call make by prefixing the var GRIDR_VENV
# egg. GRIDR_VENV="path-to-venv-target" make install
#@test -d $(basename ${GRIDR_VENV}) || mkdir -p $(basename ${GRIDR_VENV})
.PHONY: venv
venv: check ## create virtualenv in GRIDR_VENV directory if not exists
	@echo "Creating venv in $(GRIDR_VENV)"
	@test -d $(GRIDR_VENV) || python3 -m venv $(GRIDR_VENV)
	@$(GRIDR_VENV)/bin/python -m pip install $(PIP_ARG_MAIN)--no-cache-dir --upgrade pip
	@$(GRIDR_VENV)/bin/python -m pip install $(PIP_ARG_MAIN)--no-cache-dir setuptools setuptools-rust wheel build
	@$(GRIDR_VENV)/bin/python -m pip install $(PIP_ARG_MAIN)-r $(ROOT_DIR)requirements_dev.txt
	@touch $(GRIDR_VENV)/bin/activate

.PHONY: build-rust
build-rust: venv ## build the rust project
	@echo "Building libgridr.so rust library..."
	@echo "Rust crate location : $(GRIDR_RUST_CRATE_PATH)"
	@source $(GRIDR_VENV)/bin/activate && cd $(GRIDR_RUST_CRATE_PATH) && cargo build --release
	@touch $(GRIDR_LIBGRIDR_SO_BUILD_TARGET)


$(GRIDR_LIBGRIDR_SO_PYTEST_TARGET): build-rust
	rm -f "$(GRIDR_LIBGRIDR_SO_PYTEST_TARGET)"
	@ln -s $(GRIDR_LIBGRIDR_SO_BUILD_TARGET) $(GRIDR_LIBGRIDR_SO_PYTEST_TARGET)


.PHONY: build-rust-doc
build-rust-doc: venv ## build the rust project
	@echo "Building rust documentation..."
	@echo "Rust documentation location : $(GRIDR_LIBGRIDR_DOC_TARGET)"
	rm -rf $(GRIDR_LIBGRIDR_DOC_BUILD_PATH)
	@source $(GRIDR_VENV)/bin/activate && cd $(GRIDR_RUST_CRATE_PATH) && cargo doc --no-deps --document-private-items


.PHONY: build-sphinx-doc
build-sphinx-doc: venv $(GRIDR_LIBGRIDR_SO_PYTEST_TARGET) ## build the sphinx documentation
	@echo "Building sphinx documentation..."
	@echo "Sphinx documentation location : $(GRIDR_DOCS_ROOT_PATH)"
	@source $(GRIDR_VENV)/bin/activate && cd $(GRIDR_DOCS_ROOT_PATH) && make html
	tar cf $(GRIDR_DOCS_ROOT_PATH).tar $(GRIDR_DOCS_ROOT_PATH)


.PHONY: test-python
test-python: venv $(GRIDR_LIBGRIDR_SO_PYTEST_TARGET) ## perform tests on python code
	@echo "Testing..."
	@source $(GRIDR_VENV)/bin/activate && PYTHONPATH=$(ROOT_DIR)python:$(PYTHONPATH) pytest --cov=python --cov-report=xml:.coverage-reports/coverage.xml --cov-report=term --junitxml=report.xml tests/python



.PHONY: test
test: test-python ## perform all tests

.PHONY: pylint
pylint: venv ## call pylint
	@source $(GRIDR_VENV)/bin/activate && pylint $(ROOT_DIR)python --recursive=y --rcfile=$(ROOT_DIR)pylintrc_RNC2015_D  --exit-zero --halt-on-invalid-sonar-rules n > $(ROOT_DIR)pylint_report.json 



# Build package
.PHONY: build
build: venv ## build package
	@echo "Build python package"
	@$(GRIDR_VENV)/bin/python -m build





.PHONY: clean
clean: clean-venv clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

.PHONY: clean-venv
clean-venv:
	@echo "+ $@"
	@rm -rf ${GRIDR_VENV}

.PHONY: clean-build
clean-build:
	@echo "+ $@"
	@rm -fr build/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +
	@rm -fr rust/gridr/target
	@rm -f $(GRIDR_LIBGRIDR_SO_PYTEST_TARGET)

#.PHONY: clean-precommit
#clean-precommit:
#	@rm -f .git/hooks/pre-commit
#	@rm -f .git/hooks/pre-push

.PHONY: clean-pyc
clean-pyc:
	@echo "+ $@"
	@find . -type f -name "*.py[co]" -exec rm -fr {} +
	@find . -type d -name "__pycache__" -exec rm -fr {} +
	@find . -name '*~' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	@echo "+ $@"
	@rm -f .coverage
	@rm -rf .coverage.*
	@rm -rf coverage.xml
	@rm -fr htmlcov/
	@rm -fr .pytest_cache
	@rm -f pytest-report.xml
	@rm -f pylint-report.txt
	@rm -f debug.log

#ifndef GRIDR_VENV
.PHONY: print_config 
print_config: ## print configuration
	@echo "Found configuration "
	@echo " - python3 : ${PYTHON}" 
	@echo " - cargo : ${CHECK_CARGO}" 
