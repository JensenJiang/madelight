from tensorflow.python.summary.summary import SessionLog, Summary
from tensorflow.python.summary.writer.writer import FileWriter


class TensorBoardLogger:
    def __init__(self, path):
        self._writer = FileWriter(path, flush_secs=120)

    def put_start(self, global_step):
        self._writer.add_session_log(SessionLog(status=SessionLog.START), global_step)

    def put_scalar(self, k, v, global_step):
        self._writer.add_summary(Summary(value=[Summary.Value(tag=k, simple_value=float(v))]), global_step)
