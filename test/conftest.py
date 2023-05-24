import pathlib
import os

import pytest
import chimerapy as cp

# Constants
CWD = pathlib.Path(os.path.abspath(__file__)).parent
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = GIT_ROOT/'data'

@pytest.fixture
def logreceiver():
    listener = cp._logger.get_node_id_zmq_listener()
    listener.start()
    yield listener
    listener.stop()
    listener.join()
