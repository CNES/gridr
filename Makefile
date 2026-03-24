# Used environment variables
# NUMPY_VERSION : set to force numpy version
# GRIDR_SPHINX_BUILD_PATH : set sphinx output path
# BUILD_DIST_OUTDIR
# COVERAGE_REPORT_TAG
# GRIDR_VENV
# PIP_ARG_MAIN
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
ifdef GRIDR_SPHINX_BUILD_PATH
	GRIDR_SPHINX_DOC_BUILD_PATH = "$(GRIDR_SPHINX_BUILD_PATH)"
else
	GRIDR_SPHINX_DOC_BUILD_PATH = "$(GRIDR_DOCS_ROOT_PATH)/build"
endif

# Scripts path
GRIDR_SCRIPTS_PATH := $(ROOT_DIR)scripts

# Check cargo
CHECK_CARGO = $(shell command -v cargo 2> /dev/null)


# Check python3
PYTHON=$(shell command -v python3)
ifeq (, $(PYTHON))
	$(error "PYTHON=$(PYTHON) not found in $(PATH)")
endif

# Check Python version
PYTHON_VERSION_MIN = 3.10
PYTHON_VERSION_CUR = $(shell $(PYTHON) -c 'import sys; print("%d.%d"% sys.version_info[0:2])')
PYTHON_VERSION_OK = $(shell $(PYTHON) -c 'import sys; cur_ver = sys.version_info[0:2]; min_ver = tuple(map(int, "$(PYTHON_VERSION_MIN)".split("."))); print(int(cur_ver >= min_ver))')

# Extra Build configuration (abi3 feature)
BUILD_EXTRA_CONFIG := $(ROOT_DIR)dist_extra_config.cfg

NUMPY_VERSION_TAG = ""
ifndef NUMPY_VERSION
	NUMPY_VERSION_PIP = "numpy>=2.2"
else
	NUMPY_VERSION_PIP = "numpy==$(NUMPY_VERSION)"
	NUMPY_VERSION_TAG = "_numpy$(NUMPY_VERSION)"
endif

GLIBC_VERSION := $(shell ldd --version | head -n1 | grep -o '[0-9]*\.[0-9]*')
MANYLINUX_GLIBC_TAG := manylinux_2_$(shell echo $(GLIBC_VERSION) | cut -d. -f2)

ifndef BUILD_DIST_OUTDIR
	BUILD_DIST_OUTDIR = "dist_$(PYTHON_VERSION_CUR)${NUMPY_VERSION_TAG}"
endif

# Tag to use in coverage report basename
ifndef COVERAGE_REPORT_TAG
	COVERAGE_REPORT_TAG = ""
endif

# Set Virtualenv directory name
# Default if not defined is to create venv with a name that contains the current python version
# Example: GRIDR_VENV="venv/other-venv/" make install
ifndef GRIDR_VENV
	GRIDR_VENV = "$(ROOT_DIR)venv/venv_py$(PYTHON_VERSION_CUR)${NUMPY_VERSION_TAG}"
endif
GRIDR_VENV_TEST_BUILD = "${GRIDR_VENV}_test_build"

# PIP_ARG_MAIN is a variable that may contains PIP configuration arguments
# This variable may be set by the CI
ifndef PIP_ARG_MAIN
	PIP_ARG_MAIN := ""
endif

# Collect all Rust source files for dependency tracking.
# This allows Make to rebuild the library only when Rust sources actually change.
RUST_SOURCES := $(shell find $(GRIDR_RUST_CRATE_PATH)/src -name '*.rs' 2>/dev/null) \
                $(GRIDR_RUST_CRATE_PATH)/Cargo.toml \
                $(GRIDR_RUST_CRATE_PATH)/Cargo.lock

.PHONY: info
info:
	@echo "glibc version: $(GLIBC_VERSION)"
	@echo "manylinux tag: $(MANYLINUX_GLIBC_TAG)"

################ MAKE targets by sections ######################

.PHONY: help
help: ## this help
	@echo "      GRIDR MAKE HELP"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'| sort

.PHONY: check
check: ## check if cargo is installed
	@[ "$(CHECK_CARGO)" ] || ( echo ">> cargo not found"; exit 1 )

# The venv sentinel file is used to track whether the venv is up to date.
# Make compares its timestamp against requirements_dev.txt: if the requirements
# file is newer, the venv is reinstalled. This avoids the venv being recreated
# on every invocation while still reacting to dependency changes.
GRIDR_VENV_SENTINEL := $(GRIDR_VENV)/.make_sentinel

.PHONY: venv
venv: $(GRIDR_VENV_SENTINEL) ## create virtualenv in GRIDR_VENV directory if not exists

$(GRIDR_VENV_SENTINEL): $(ROOT_DIR)requirements_dev.txt
	@if [ "$(PYTHON_VERSION_OK)" = "0" ] ; then \
		echo "WARNING: Python $(PYTHON_VERSION_CUR) < $(PYTHON_VERSION_MIN) required"; \
	fi
	@echo "Creating/updating venv in $(GRIDR_VENV)"
	@test -d $(GRIDR_VENV) || python3 -m venv $(GRIDR_VENV)
	@$(GRIDR_VENV)/bin/python -m pip install $(PIP_ARG_MAIN)--no-cache-dir --upgrade pip
	@$(GRIDR_VENV)/bin/python -m pip install $(PIP_ARG_MAIN)--no-cache-dir setuptools setuptools-rust wheel build
	@$(GRIDR_VENV)/bin/python -m pip install $(PIP_ARG_MAIN)--no-cache-dir ${NUMPY_VERSION_PIP}
	@$(GRIDR_VENV)/bin/python -m pip install $(PIP_ARG_MAIN)-r $(ROOT_DIR)requirements_dev.txt
	@touch $(GRIDR_VENV_SENTINEL)

# The compiled .so is the real build artifact. Make tracks it against all Rust
# sources: a rebuild is triggered only when a source file is newer than the .so,
# or when the .so is absent. The symlink for pytest is created here as well,
# so there is a single rule responsible for both the build and the link.
#
# FIX — previously build-rust was .PHONY, which forced a Rust rebuild on every
# invocation of any target that depended on it (e.g. test-python). Removing
# .PHONY and making the .so the target lets Make decide whether work is needed.
$(GRIDR_LIBGRIDR_SO_BUILD_TARGET): $(GRIDR_VENV_SENTINEL) $(RUST_SOURCES) | check
	@echo "Building libgridr.so rust library..."
	@echo "Rust crate location : $(GRIDR_RUST_CRATE_PATH)"
	@source $(GRIDR_VENV)/bin/activate && cd $(GRIDR_RUST_CRATE_PATH) && cargo build --release

# The pytest symlink depends only on the compiled .so.
# It is recreated whenever the .so is rebuilt (i.e. newer than the symlink).
$(GRIDR_LIBGRIDR_SO_PYTEST_TARGET): $(GRIDR_LIBGRIDR_SO_BUILD_TARGET)
	@echo "Updating pytest symlink..."
	@rm -f "$(GRIDR_LIBGRIDR_SO_PYTEST_TARGET)"
	@ln -s $(GRIDR_LIBGRIDR_SO_BUILD_TARGET) $(GRIDR_LIBGRIDR_SO_PYTEST_TARGET)

# Convenience phony alias so callers can still run `make build-rust`
.PHONY: build-rust
build-rust: $(GRIDR_LIBGRIDR_SO_BUILD_TARGET) ## build the rust project

.PHONY: test-rust
test-rust: $(GRIDR_VENV_SENTINEL) ## test rust code
	@echo "Test rust code..."
	@source $(GRIDR_VENV)/bin/activate && cd $(GRIDR_RUST_CRATE_PATH) && RUST_BACKTRACE=1 cargo test -- --nocapture

.PHONY: build-rust-doc
build-rust-doc: $(GRIDR_VENV_SENTINEL) ## build the rust documentation
	@echo "Building rust documentation..."
	@echo "Rust documentation location : $(GRIDR_LIBGRIDR_DOC_BUILD_PATH)"
	rm -rf $(GRIDR_LIBGRIDR_DOC_BUILD_PATH)
	@source $(GRIDR_VENV)/bin/activate && cd $(GRIDR_RUST_CRATE_PATH) && cargo doc --no-deps --document-private-items


.PHONY: build-sphinx-doc
build-sphinx-doc: $(GRIDR_VENV_SENTINEL) $(GRIDR_LIBGRIDR_SO_PYTEST_TARGET) clean-sphinx-doc ## build the sphinx documentation
	@echo "Building sphinx documentation..."
	@echo "Sphinx documentation location : $(GRIDR_DOCS_ROOT_PATH)"
	@source $(GRIDR_VENV)/bin/activate && sphinx-build -M html $(GRIDR_DOCS_ROOT_PATH)/source $(GRIDR_SPHINX_DOC_BUILD_PATH)


.PHONY: test-python
test-python: $(GRIDR_VENV_SENTINEL) $(GRIDR_LIBGRIDR_SO_PYTEST_TARGET) ## perform tests on python code
	@echo "Testing..."
	@source $(GRIDR_VENV)/bin/activate && PYTHONPATH=$(ROOT_DIR)python:$(PYTHONPATH) pytest --cov=python --cov-report=xml:.coverage-reports/coverage$(COVERAGE_REPORT_TAG).xml --cov-report=term --junitxml=report$(COVERAGE_REPORT_TAG).xml tests/python --ignore=tests/python/benchmarks


.PHONY: test
test: test-rust test-python ## perform all tests

.PHONY: pylint
pylint: $(GRIDR_VENV_SENTINEL) ## call pylint
	@source $(GRIDR_VENV)/bin/activate && pylint $(ROOT_DIR)python --recursive=y --rcfile=$(ROOT_DIR)pylintrc_RNC2015_D  --exit-zero --halt-on-invalid-sonar-rules n > $(ROOT_DIR)pylint_report.json


# Build package
.PHONY: build
build: $(GRIDR_VENV_SENTINEL) clean-build ## build package
	@echo "Build python package"
	@DIST_EXTRA_CONFIG=${BUILD_EXTRA_CONFIG} $(GRIDR_VENV)/bin/python -m build --outdir $(BUILD_DIST_OUTDIR)
	@echo "Set wheel for manylinux base on glibc version"
	@$(GRIDR_VENV)/bin/auditwheel repair $(BUILD_DIST_OUTDIR)/*.whl --plat $(MANYLINUX_GLIBC_TAG)_x86_64 -w $(BUILD_DIST_OUTDIR)_fixed/

# Create NOTICE
.PHONY: check-licenses
check-licenses: build ## check licenses and generate NOTICE
	@echo "Check licenses"
	@echo "Create isolated venv to install built package"
	@test -d $(GRIDR_VENV_TEST_BUILD) || rm -rf $(GRIDR_VENV_TEST_BUILD)
	@python3 -m venv $(GRIDR_VENV_TEST_BUILD)
	@$(GRIDR_VENV_TEST_BUILD)/bin/python -m pip install $(PIP_ARG_MAIN)--no-cache-dir --upgrade pip
	@$(GRIDR_VENV_TEST_BUILD)/bin/python -m pip install $(PIP_ARG_MAIN)--no-cache-dir pip-licenses-lib
	@$(GRIDR_VENV_TEST_BUILD)/bin/python -m pip install $(PIP_ARG_MAIN)--no-cache-dir $(BUILD_DIST_OUTDIR)_fixed/gridr*.whl
	@$(GRIDR_VENV_TEST_BUILD)/bin/python $(GRIDR_SCRIPTS_PATH)/generate_notice.py

.PHONY: clean
clean: clean-venv clean-build clean-pyc clean-test clean-sphinx-doc ## remove all build, test, coverage and Python artifacts

.PHONY: clean-venv
clean-venv:
	@echo "+ $@"
	@rm -rf ${GRIDR_VENV}
	@rm -rf $(GRIDR_VENV_TEST_BUILD)

.PHONY: clean-build
clean-build:
	@echo "+ $@"
	@rm -fr dist*/
	@rm -fr build*/
	@rm -fr .eggs/
	@find . -name '*.egg-info' -exec rm -fr {} +
	@find . -name '*.egg' -exec rm -f {} +
	@rm -fr rust/gridr/target
	@rm -f $(GRIDR_LIBGRIDR_SO_PYTEST_TARGET)

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

.PHONY: clean-sphinx-doc
clean-sphinx-doc:
	@echo "+ $@"
	@rm -rf $(GRIDR_SPHINX_DOC_BUILD_PATH)
	@rm -rf $(GRIDR_DOCS_ROOT_PATH)/source/_notebooks/generated

.PHONY: print_config
print_config: ## print configuration
	@echo "Found configuration "
	@echo " - python3 : ${PYTHON}"
	@echo " - cargo : ${CHECK_CARGO}"
