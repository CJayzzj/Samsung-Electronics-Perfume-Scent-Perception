# -*- coding: utf-8 -*-
"""机器人端原始传感器帧保存模块。

功能：使用后台线程把每一帧写入CSV或Excel，达到设定行数后自动分卷，
避免磁盘写入阻塞实时采集，也避免单个数据文件过大。
"""

import csv
import os
import queue
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


# CSV和Excel使用相同列顺序，便于后续训练程序直接读取。
COLUMNS = (
    ["timestamp_ms", "device_id", "sequence"]
    + ["Resistance_Sensor%d" % index for index in range(1, 17)]
    + ["Temperature", "Humidity"]
    + ["Recognition_Result"]
)
RECOGNITION_RESULT_KEY = "recognition_result"
PHASE_RESULTS = {
    "warmup": "预热中",
    "idle": "等待信号",
    "rising": "响应上升",
    "falling": "识别中",
    "recovering": "恢复中",
}
_STOP = object()


def frame_row(frame):
    """把协议字典转换成适合写入表格的一行数据。"""
    return (
        [frame["timestamp_ms"], frame["device_id"], frame["sequence"]]
        + list(frame["resistance"])
        + [frame["temperature_c"], frame["humidity_rh"]]
        + [str(frame.get(RECOGNITION_RESULT_KEY) or "未收到推理状态")]
    )


class InferenceFrameJoiner:
    """按设备和帧序号合并原始帧、推理状态及最终标签后再写盘。"""

    PRIORITY_FALLBACK = 0
    PRIORITY_STATUS = 1
    PRIORITY_EVENT = 2
    PRIORITY_LABEL = 3

    def __init__(self, recorder, max_pending=1000, on_warning=None):
        """保存后台记录器，并创建线程安全的待匹配帧缓存。"""
        self.recorder = recorder
        self.max_pending = max(1, int(max_pending))
        self.on_warning = on_warning
        self._pending = OrderedDict()
        self._timestamp_index = {}
        self._lock = threading.Lock()

    @staticmethod
    def _frame_key(frame):
        """设备编号和sequence共同组成唯一帧键。"""
        return str(frame["device_id"]), int(frame["sequence"])

    @staticmethod
    def _timestamp_key(frame):
        """恢复消息没有sequence时，使用同设备时间戳关联当前帧。"""
        return str(frame["device_id"]), int(frame["timestamp_ms"])

    def _remove_locked(self, key):
        """在持锁状态下移除一帧及其时间戳索引。"""
        entry = self._pending.pop(key)
        timestamp_key = self._timestamp_key(entry["frame"])
        if self._timestamp_index.get(timestamp_key) == key:
            self._timestamp_index.pop(timestamp_key, None)
        return entry

    def _drain_ready_locked(self):
        """只按接收顺序取出队首连续就绪帧，避免文件sequence乱序。"""
        ready = []
        while self._pending:
            first_key = next(iter(self._pending))
            if not self._pending[first_key]["ready"]:
                break
            ready.append(self._remove_locked(first_key))
        return ready

    def _write_entries(self, entries):
        """在锁外把已完成匹配的帧交给异步保存线程。"""
        for entry in entries:
            result = entry["result"] or "未收到推理状态"
            self.recorder.record(entry["frame"], result)

    def _warn(self, message):
        """缓存异常时通过主程序工业日志输出WARN。"""
        if callable(self.on_warning):
            self.on_warning(str(message))

    def add_frame(self, frame):
        """发送推理前暂存一帧；缓存超限时保底写入最早帧。"""
        copied = dict(frame)
        copied["resistance"] = list(frame["resistance"])
        key = self._frame_key(copied)
        overflow = []
        with self._lock:
            if key in self._pending:
                raise ValueError(
                    "duplicate pending frame: device_id=%s sequence=%d" % key
                )
            self._pending[key] = {
                "frame": copied,
                "result": None,
                "priority": -1,
                "ready": False,
            }
            self._timestamp_index[self._timestamp_key(copied)] = key
            while len(self._pending) > self.max_pending:
                oldest_key = next(iter(self._pending))
                entry = self._remove_locked(oldest_key)
                if entry["priority"] < self.PRIORITY_FALLBACK:
                    entry["result"] = "未收到推理状态"
                    entry["priority"] = self.PRIORITY_FALLBACK
                entry["ready"] = True
                overflow.append(entry)
        if overflow:
            self._write_entries(overflow)
            self._warn(
                "推理状态待匹配帧超过%d，已保底保存%d帧"
                % (self.max_pending, len(overflow))
            )

    def _find_sequence_key_locked(self, payload, field_name):
        """按消息中的设备和sequence查找帧；错误消息可省略设备号。"""
        value = payload.get(field_name)
        if value is None:
            return None
        sequence = int(value)
        device_id = str(payload.get("device_id") or "").strip()
        if device_id:
            key = (device_id, sequence)
            return key if key in self._pending else None
        matches = [key for key in self._pending if key[1] == sequence]
        return matches[0] if len(matches) == 1 else None

    def _annotate(self, key, text, priority, ready=False):
        """按优先级更新一帧，并在status/error到达后顺序写出。"""
        entries = []
        found = False
        with self._lock:
            entry = self._pending.get(key)
            if entry is not None:
                found = True
                if int(priority) >= int(entry["priority"]):
                    entry["result"] = str(text)
                    entry["priority"] = int(priority)
                if ready:
                    entry["ready"] = True
                entries = self._drain_ready_locked()
        self._write_entries(entries)
        return found

    def handle_status(self, payload):
        """普通status提供同帧阶段，也是正常帧完成匹配的提交信号。"""
        with self._lock:
            key = self._find_sequence_key_locked(payload, "sequence")
        if key is None:
            return False
        result = PHASE_RESULTS.get(str(payload.get("phase") or ""), "推理状态未知")
        return self._annotate(key, result, self.PRIORITY_STATUS, ready=True)

    def handle_result(self, payload):
        """把最终label写到confirm_sequence，等待随后status统一提交。"""
        with self._lock:
            key = self._find_sequence_key_locked(payload, "confirm_sequence")
        if key is None:
            return False
        label = str(payload.get("label") or "").strip() or "推理错误"
        return self._annotate(key, label, self.PRIORITY_LABEL, ready=False)

    def handle_recovery(self, payload):
        """按source_timestamp_ms标记恢复完成或超时，优先于普通状态。"""
        timestamp_ms = payload.get("source_timestamp_ms")
        device_id = str(payload.get("device_id") or "").strip()
        if timestamp_ms is None or not device_id:
            return False
        with self._lock:
            key = self._timestamp_index.get((device_id, int(timestamp_ms)))
        if key is None:
            return False
        result = "恢复完成" if str(payload.get("status") or "") == "complete" else "恢复超时"
        return self._annotate(key, result, self.PRIORITY_EVENT, ready=False)

    def handle_error(self, payload):
        """单帧推理错误通常没有后续status，因此标记后立即允许写盘。"""
        with self._lock:
            key = self._find_sequence_key_locked(payload, "source_sequence")
        if key is None:
            return False
        return self._annotate(key, "推理错误", self.PRIORITY_EVENT, ready=True)

    def flush_unresolved(self):
        """退出时保留所有原始帧；已有高优先级结果不被兜底文本覆盖。"""
        with self._lock:
            count = len(self._pending)
            for entry in self._pending.values():
                if entry["priority"] < self.PRIORITY_FALLBACK:
                    entry["result"] = "未收到推理状态"
                    entry["priority"] = self.PRIORITY_FALLBACK
                entry["ready"] = True
            entries = self._drain_ready_locked()
        self._write_entries(entries)
        return count

    @property
    def pending_count(self):
        """返回当前等待推理消息的帧数，供测试和退出日志使用。"""
        with self._lock:
            return len(self._pending)


class AsyncFrameRecorder:
    """异步保存每一帧原始数据，并在文件过大前自动分卷。"""

    def __init__(
        self,
        output_dir,
        output_format="csv",
        rotate_rows=50000,
        xlsx_flush_every=20,
        queue_size=10000,
    ):
        """创建输出目录、写入队列和后台保存线程。"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_format = str(output_format).lower()
        if self.output_format not in {"csv", "xlsx"}:
            raise ValueError("output_format must be csv or xlsx")
        self.rotate_rows = max(1, int(rotate_rows))
        self.xlsx_flush_every = max(1, int(xlsx_flush_every))
        # 有界队列限制待写数据量；队列满时record会等待，防止静默丢帧。
        self.queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.error = None
        self.rows_written = 0
        self.current_path = None
        self._part = 0
        # 文件操作放入独立线程，主线程可以继续接收和转发电子鼻数据。
        self._thread = threading.Thread(
            target=self._run,
            name="raw-frame-recorder",
            daemon=True,
        )
        self._thread.start()

    def record(self, frame, recognition_result=None):
        """复制一帧并放入保存队列；后台异常会立即反馈给主程序。"""
        if self.error is not None:
            raise RuntimeError("raw data recorder failed: %s" % self.error)
        item = dict(frame)
        if recognition_result is not None:
            item[RECOGNITION_RESULT_KEY] = str(recognition_result)
        self.queue.put(item)

    def close(self, timeout_s=30.0):
        """通知后台线程写完剩余数据，并等待其安全停止。"""
        # 特殊_STOP对象排在已有帧之后，可保证关闭前先写完队列数据。
        self.queue.put(_STOP)
        self._thread.join(float(timeout_s))
        if self._thread.is_alive():
            raise TimeoutError("raw data recorder did not stop in time")
        if self.error is not None:
            raise RuntimeError("raw data recorder failed: %s" % self.error)

    def _next_path(self):
        """按当前时间、分卷序号和格式生成下一个输出文件名。"""
        self._part += 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = ".csv" if self.output_format == "csv" else ".xlsx"
        return self.output_dir / ("enose_raw_%s_part%03d%s" % (stamp, self._part, suffix))

    def _run(self):
        """后台线程入口：按输出格式选择CSV或Excel保存流程。"""
        try:
            if self.output_format == "csv":
                self._run_csv()
            else:
                self._run_xlsx()
        except Exception as exc:
            self.error = exc

    def _run_csv(self):
        """持续写CSV，每帧刷新磁盘，并在达到行数上限时新建分卷。"""
        stream = None
        writer = None
        rows_in_part = 0
        try:
            while True:
                item = self.queue.get()
                if item is _STOP:
                    break
                # 第一次写入或当前分卷已满时，关闭旧文件并写入新表头。
                if stream is None or rows_in_part >= self.rotate_rows:
                    if stream is not None:
                        stream.close()
                    self.current_path = self._next_path()
                    stream = open(self.current_path, "w", newline="", encoding="utf-8-sig")
                    writer = csv.writer(stream)
                    writer.writerow(COLUMNS)
                    rows_in_part = 0
                # 每帧flush，异常断电时尽可能保留已经采集的数据。
                writer.writerow(frame_row(item))
                stream.flush()
                rows_in_part += 1
                self.rows_written += 1
        finally:
            if stream is not None:
                stream.close()

    def _run_xlsx(self):
        """分批写Excel，并通过原子替换降低文件损坏和保存失败风险。"""
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            raise RuntimeError("xlsx output requires openpyxl") from exc

        workbook = None
        worksheet = None
        rows_in_part = 0
        unsaved_rows = 0
        while True:
            item = self.queue.get()
            if item is _STOP:
                # 退出前补保存尚未达到定期刷新数量的最后几行。
                if workbook is not None and unsaved_rows:
                    self._atomic_save_workbook(workbook, self.current_path)
                break
            # 新分卷先保存旧工作簿，再创建带统一表头的新工作簿。
            if workbook is None or rows_in_part >= self.rotate_rows:
                if workbook is not None and unsaved_rows:
                    self._atomic_save_workbook(workbook, self.current_path)
                self.current_path = self._next_path()
                workbook = Workbook()
                worksheet = workbook.active
                worksheet.title = "raw_data"
                worksheet.append(COLUMNS)
                rows_in_part = 0
                unsaved_rows = 0
            worksheet.append(frame_row(item))
            rows_in_part += 1
            unsaved_rows += 1
            self.rows_written += 1
            # Excel整体保存开销较大，因此累计指定行数后再刷新一次。
            if unsaved_rows >= self.xlsx_flush_every:
                self._atomic_save_workbook(workbook, self.current_path)
                unsaved_rows = 0

    @staticmethod
    def _atomic_save_workbook(workbook, target_path):
        """先保存临时Excel，再原子替换目标文件，避免留下半写文件。"""
        target_path = Path(target_path)
        temporary = target_path.with_name(target_path.stem + ".tmp.xlsx")
        workbook.save(temporary)
        os.replace(str(temporary), str(target_path))
