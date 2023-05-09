from typing import Optional, Union

import time
import datetime
import pandas as pd
import chimerapy as cp
from chimerapy_orchestrator import source_node

@source_node(name="MMLAPIPE_LogReader")
class LogReader(cp.Node):
    """A node that processes logs from elp package to obtain game state information.

    Parameters
    ----------
    name: str, optional (default: "LogReader")
        The name of the node
    data_key: str, optional (default: "data")
        The key to use for the frame in the data chunk
    **kwargs
        Additional keyword arguments to pass to the Node constructor
    """

    def __init__(
        self,
        logfile: str,
        timestamp_column: str = "timestamp",
        timestamp_format: Optional[str] = None,
        offset: Optional[Union[datetime.datetime, float]] = None,
        name: str = "LogReader",
        data_key: str = "data",
        **kwargs,
    ) -> None:

        self.logfile = logfile
        self.timestamp_column = timestamp_column
        self.timestamp_format = timestamp_format
        self.offset = offset
        self.data_key = data_key
        self.row_id: int = 0
        self.debug = False
        super().__init__(name=name, **kwargs)

    def setup(self) -> None:

        # Read file
        if self.logfile.endswith('.csv'):
            self.data = pd.read_csv(self.logfile)
        elif self.logfile.endswith('.xlsx'):
            self.data = pd.read_excel(self.logfile)
        else:
            raise NotImplementedError(f"{self.logfile}, file not supported by LogReader")

        # Convert string timestamps to datetime objects
        if self.timestamp_format:
            self.data[self.timestamp_column] = pd.to_datetime(self.data[self.timestamp_column], format=self.timestamp_format)
        else:
            self.data[self.timestamp_column] = pd.to_datetime(self.data[self.timestamp_column])

        # Apply offset if requested
        if type(self.offset) != type(None):
           
            # Apply second type offset
            if isinstance(self.offset, (float, int)):
                self.data[self.timestamp_column] = (self.data[self.timestamp_column] - self.data[self.timestamp_column][0]).dt.total_seconds()
                self.data = self.data[self.data[self.timestamp_column] >= self.offset]
                self.data[self.timestamp_column] = self.data[self.timestamp_column] - self.offset

            # Apply datetime offset
            elif isinstance(self.offset, datetime.datetime):
                self.data = self.data[self.data[self.timestamp_column] >= self.offset]
                self.data[self.timestamp_column] = self.data[self.timestamp_column] - self.offset

        self.row_id = 0

    def step(self) -> cp.DataChunk:
        data_chunk = cp.DataChunk()

        # For first, send ASAP
        if self.row_id == 0:
            data_chunk.add(self.data_key, self.data.iloc[self.row_id])
        else:
            current_time = self.data[self.timestamp_column].iloc[self.row_id-1]
            next_time = self.data[self.timestamp_column].iloc[self.row_id]

            # Get delta and sleep that (make sure its in seconds)
            delta = next_time - current_time
            if not isinstance(delta, (int, float)):
                delta = delta.total_seconds()
            time.sleep(max(delta, 0))

            # Then add chunk and send
            data_chunk.add(self.data_key, self.data.iloc[self.row_id])

        # Update
        self.row_id += 1

        if self.debug:
            self.logger.debug(data_chunk.get(self.data_key)['value'].to_dict())

        return data_chunk
