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
        batch_window_size: Union[int, float] = 0.25,
        timestamp_column: str = "timestamp",
        timestamp_format: Optional[str] = None,
        offset: Optional[Union[datetime.datetime, float]] = None,
        sleep_factor: float = 0.95,
        name: str = "LogReader",
        data_key: str = "data",
        **kwargs,
    ) -> None:

        self.logfile = logfile
        self.batch_window_size = batch_window_size
        self.timestamp_column = timestamp_column
        self.timestamp_format = timestamp_format
        self.offset = offset
        self.data_key = data_key
        self.sleep_factor = sleep_factor

        self.first_pass = True
        self.step_id = 0
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
                self.data[self.timestamp_column] = (self.data[self.timestamp_column] - self.offset).dt.total_seconds()

        # Make copy of data as a stack
        self.stack = self.data.copy()
        
        self.first_pass = True
        self.step_id = 0

    def step(self) -> cp.DataChunk:
        data_chunk = cp.DataChunk()

        # Get initial time
        if self.step_id == 0:
            self.initial = datetime.datetime.now()
        
        # Compute the current datetime
        next_timestamp = (self.step_id + 1) * self.batch_window_size

        # Trim stack
        stop_id = 0
        for i, (_, row) in enumerate(self.stack.iterrows()):
            if row[self.timestamp_column] >= next_timestamp:
                stop_id = i
                break

        selected_data = self.stack.iloc[:stop_id]
        data_chunk.add(self.data_key, selected_data)
        self.stack = self.stack.iloc[stop_id:]
 
        current_datetime = datetime.datetime.now()
        delta = (current_datetime - self.initial).total_seconds()
        sleep_time = max(next_timestamp - delta, 0)
        time.sleep(sleep_time*self.sleep_factor)
        
        # Update
        self.step_id += 1

        return data_chunk
