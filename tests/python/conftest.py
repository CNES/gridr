from pathlib import Path

import pytest

OUTPUT_DIR = Path("./tests_data/out/")


@pytest.fixture
def tests_data_out_dir():
    """Tests data output directory"""
    yield OUTPUT_DIR
