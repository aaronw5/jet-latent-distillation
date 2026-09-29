import pytest
from jetdistill import config



@pytest.fixture(autouse=True)
def need_data():
    if not (config.DATA / 'splits.npz').exists():
        pytest.skip('dataset files not found (set JETDISTILL_DATA)')
