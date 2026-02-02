import logging
import threading
from queue import Queue
from typing import Optional


class LogQueueManager:
    """Manages a logging queue and worker thread for multiprocessing logging.
    
    This class handles the setup and teardown of a queue-based logging system
    suitable for multiprocessing scenarios where worker processes need to send
    log records to a central logger in the main process.
    
    Usage:
        with LogQueueManager(logger, mp_manager) as log_mgr:
            # Pass log_mgr.queue to worker processes
            pool = ProcessPoolExecutor(
                initializer=worker_init,
                initargs=(log_mgr.queue,)
            )
            # ... do work ...
    """

    def __init__(self,
                 logger: logging.Logger,
                 mp_manager=None,
                 queue: Optional[Queue] = None,
                 ):
        """Initialize the log queue manager.
        
        Args:
            logger: The logger that will handle records from the queue.
            mp_manager: Optional multiprocessing Manager for creating queue.
                       If None and queue is None, creates a regular Queue.
            queue: Optional existing queue to use. If None, creates a new one.
        """
        self.logger = logger

        # Create or use provided queue
        if queue is not None:
            self._queue = queue
        elif mp_manager is not None:
            self._queue = mp_manager.Queue()
        else:
            self._queue = Queue()

        # Internal state
        self._thread: Optional[threading.Thread] = None
        self._closed = False

        # Create and start the logger thread
        self._thread = threading.Thread(
            target=self._logger_worker,
            args=(self.logger, self._queue),
            daemon=False,
            name="LogQueueWorker",
        )
        self._thread.start()
        self.logger.debug("Started LogQueueManager")

    @property
    def queue(self) -> Queue:
        """Get the logging queue for worker processes."""
        return self._queue

    @staticmethod
    def _logger_worker(logger: logging.Logger, q: Queue):
        """Worker function that processes log records from the queue.
        
        Args:
            logger: Logger to handle records.
            q: Queue to receive records from.
        """
        while True:
            record = q.get()
            if record is None:  # Sentinel
                break
            logger.handle(record)

    def close(self, timeout: float = 5.0):
        """Close the log queue manager and wait for thread to finish.
        
        Args:
            timeout: Maximum seconds to wait for thread to finish.
            
        Raises:
            RuntimeError: If thread does not finish in time.
        """
        if self._closed:
            self.logger.warning("LogQueueManager already closed")
            return

        self.logger.debug("Closing LogQueueManager...")
        self._queue.put(None)  # Sentinel

        if self._thread is not None:
            self._thread.join(timeout=timeout)

            if self._thread.is_alive():
                self.logger.error("Logger thread did not finish in time!")
                raise RuntimeError("Logger thread timeout")

        self._closed = True
        self.logger.debug("Closed LogQueueManager")

    def is_alive(self) -> bool:
        """Check if the logger thread is still running.
        
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
