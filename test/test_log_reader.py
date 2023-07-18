# Import Built-in
import logging
import time

# Internal Imports
from chimerapy.pipelines.generic_nodes.log_reader import LogReader

# Constants
from .conftest import DATA_DIR

LOGFILE = DATA_DIR / "embodied" / "fish-only-logs.csv"

logger = logging.getLogger()


def test_log_reader_main(logreceiver):

    log_reader = LogReader(
        logfile=str(LOGFILE),
        data_key="data",
        timestamp_column="timestamp",
        timestamp_format="%H:%M:%S:%f",
        offset=518.5,
        name="logs",
        debug_port=logreceiver.port,
    )

    log_reader.run(blocking=False)
    logger.info("Outside of Node execution")

    time.sleep(5)

    logger.info("Shutting down Node")
    log_reader.shutdown()
