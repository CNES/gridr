from pathlib import Path

import pytest


@pytest.fixture
def original_datadir(request):
    # conftest dir
    conftest_dir = Path(__file__).parent

    # current test path
    test_file_path = Path(request.fspath)
    test_dir = test_file_path.parent

    # get relative filename
    try:
        relative_path = test_dir.relative_to(conftest_dir)
    except ValueError:
        relative_path = Path(".")

    base_dir = conftest_dir / "_regression_data" / relative_path / test_file_path.stem

    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n --> regression data directory : {base_dir}")
    return base_dir


@pytest.fixture
def lazy_datadir(original_datadir):
    return original_datadir
