import os
import pathlib

import pytest

import chimerapy.engine as cpe

# Constants
CWD = pathlib.Path(os.path.abspath(__file__)).parent
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = GIT_ROOT / "data"


@pytest.fixture
def logreceiver():
    listener = cpe._logger.get_node_id_zmq_listener()
    listener.start()
    yield listener
    listener.stop()
    listener.join()
