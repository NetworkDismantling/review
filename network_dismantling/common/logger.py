# class MultiprocessingLogger(Logger):
#     _queue: Queue = None
#
#     def __init__(self, queue: Queue):
#         self._queue = queue
#
#


def logger_thread(logger, q):
    while True:
        record = q.get()
        if record is None:
            break
        # logger = logging.getLogger(record.name)
        logger.handle(record)
