import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, List, Optional

import pandas as pd


class BaseDataFrameWriter(ABC):
    """Abstract base class for threaded DataFrame writers.
    
    Provides common functionality for CSV and Parquet writers:
    - Thread lifecycle management (start, close, is_alive)
    - Context manager support
    - Thread state validation
    - Queue-based communication
    
    Subclasses must implement:
    - _create_writer_thread(): Create the actual worker thread
    """

    def __init__(self,
                 output_file: Union[Path, str],
                 columns: Union[str, List[str]],
                 logger: logging.Logger = logging.getLogger("dummy"),
                 ):
        """Initialize the DataFrame writer.
        
        Args:
            output_file: Path to output file.
            columns: Column names to write.
            logger: Logger for messages.
        """
        self.output_file = Path(output_file) if not isinstance(output_file, Path) else output_file
        self.columns = columns if isinstance(columns, list) else [columns]
        self.logger = logger

        # Internal state
        self._closed = False
        self._thread: Optional[threading.Thread] = None

        # Create and start writer thread (implemented by subclass)
        self._thread = self._create_writer_thread()
        self._thread.start()
        self.logger.info(f"Started {self.__class__.__name__} for {self.output_file}")

    @abstractmethod
    def _create_writer_thread(self) -> threading.Thread:
        """Create the writer thread.
        
        Subclasses must implement this to create a threading.Thread
        configured with the appropriate target function and arguments.
        
        Returns:
            Configured Thread object (not started).
        """
        pass

    @abstractmethod
    def write(self, df: pd.DataFrame):
        """Write a DataFrame to the output file.
        
        Args:
            df: DataFrame to write.
            
        Raises:
            ValueError: If writer is already closed.
            RuntimeError: If writer thread has died.
        """
        pass

    def close(self, timeout: float = 30.0):
        """Close the writer and wait for all data to be written.
        
        Args:
            timeout: Maximum seconds to wait for thread to finish.
            
        Raises:
            RuntimeError: If thread does not finish in time.
        """
        if self._closed:
            self.logger.warning(f"{self.__class__.__name__} already closed")
            return

        self.logger.debug(f"Closing {self.__class__.__name__}...")
        self._send_sentinel()

        if self._thread is not None:
            self._thread.join(timeout=timeout)

            if self._thread.is_alive():
                self.logger.error("Writer thread did not finish in time!")
                raise RuntimeError("Writer thread timeout")

        self._closed = True
        self.logger.info(f"Closed {self.__class__.__name__} for {self.output_file}")

    @abstractmethod
    def _send_sentinel(self):
        """Send sentinel value to stop the writer thread."""
        pass

    def is_alive(self) -> bool:
        """Check if the writer thread is still running.
        
        Returns:
            True if thread is alive, False otherwise.
        """
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures close is called."""
        self.close()
        return False  # Don't suppress exceptions
