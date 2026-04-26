from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

_rt_logger = logging.getLogger(__name__)

import yaml

try:
    import pylsl
except ImportError:
    pylsl = None  # type: ignore[assignment]

from psychopy import core, event, visual

from markers import *
from experiment_config import *



# 事件日志记录器 —— 将实验事件以 CSV 格式持久化到磁盘
class EventLogger:
    def __init__(self, output_path: Path, config: ExperimentConfig) -> None:
        self.output_path = output_path
        self.config = config
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output_path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=[
                "event_type",
                "monotonic_s",
                "unix_time_s",
                "participant",
                "session",
                "run",
                "mode",
                "block_index",
                "trial_index_in_block",
                "global_trial_index",
                "condition",
                "trial_type",
                "phase_name",
                "marker_name",
                "marker_value",
                "ssvep_target_side",
                "ssvep_target_freq_hz",
                "note",
            ],
        )
        self.writer.writeheader()
        self.file.flush()

    def log_event(
        self,
        event_type: str,
        *,
        block_index: int | str = "",
        trial_index_in_block: int | str = "",
        global_trial_index: int | str = "",
        condition: str = "",
        trial_type: str = "",
        phase_name: str = "",
        marker_name: str = "",
        marker_value: int | float | str = "",
        ssvep_target_side: str = "",
        ssvep_target_freq_hz: float | str = "",
        note: str = "",
    ) -> None:
        row = {
            "event_type": event_type,
            "monotonic_s": f"{time.perf_counter():.6f}",
            "unix_time_s": f"{time.time():.6f}",
            "participant": self.config.participant,
            "session": self.config.session,
            "run": self.config.run,
            "mode": self.config.session_cfg.mode,
            "block_index": block_index,
            "trial_index_in_block": trial_index_in_block,
            "global_trial_index": global_trial_index,
            "condition": condition,
            "trial_type": trial_type,
            "phase_name": phase_name,
            "marker_name": marker_name,
            "marker_value": marker_value,
            "ssvep_target_side": ssvep_target_side,
            "ssvep_target_freq_hz": ssvep_target_freq_hz,
            "note": note,
        }
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


# UDP marker 发送器 —— 以 big-endian float 格式发送 marker 到 OpenBCI GUI
class UdpMarkerSender:
    """以 big-endian 32 位浮点格式发送 marker，兼容 OpenBCI GUI 的 Marker UDP 输入。"""

    def __init__(self, network: NetworkConfig, logger: EventLogger) -> None:
        self.address = (network.udp_ip, network.udp_port)
        self.logger = logger
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(
        self,
        marker_value: int,
        marker_name: str,
        *,
        block_index: int | str = "",
        trial_index_in_block: int | str = "",
        global_trial_index: int | str = "",
        condition: str = "",
        trial_type: str = "",
        phase_name: str = "",
        ssvep_target_side: str = "",
        ssvep_target_freq_hz: float | str = "",
        note: str = "",
    ) -> None:
        payload = struct.pack(">f", float(marker_value))
        self.sock.sendto(payload, self.address)
        self.logger.log_event(
            "marker",
            block_index=block_index,
            trial_index_in_block=trial_index_in_block,
            global_trial_index=global_trial_index,
            condition=condition,
            trial_type=trial_type,
            phase_name=phase_name,
            marker_name=marker_name,
            marker_value=marker_value,
            ssvep_target_side=ssvep_target_side,
            ssvep_target_freq_hz=ssvep_target_freq_hz,
            note=note,
        )

    def close(self) -> None:
        self.sock.close()


# LabRecorderCLI 子进程管理器 —— 后台启动/停止 XDF 录制
class LabRecorderCLIController:
    """管理 LabRecorderCLI 作为后台子进程进行 XDF 录制。

    LabRecorderCLI 用法::

        LabRecorderCLI.exe outputfile.xdf ''searchstr'' [''searchstr2'' ...]

    该子进程将所有匹配的 LSL 流录制到单个 XDF 文件中。
    录制将持续到进程被终止为止。
    """

    def __init__(self, config: LabRecorderConfig) -> None:
        self.config = config
        self.process: subprocess.Popen | None = None
        self._xdf_path: str = ""
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def _read_stderr(self) -> None:
        """后台线程，用于读取 LabRecorderCLI 的 stderr 输出。"""
        if self.process is None or self.process.stderr is None:
            return
        try:
            for line in self.process.stderr:
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    self._stderr_lines.append(decoded)
        except Exception:
            pass  # 进程已终止，管道已关闭

    def get_stderr_log(self) -> str:
        """返回 LabRecorderCLI 累积的 stderr 输出。"""
        return "\n".join(self._stderr_lines)

    def resolve_xdf_path(self, participant: str, session: str,
                         run: int, task: str, class_mode: str = "binary") -> str:
        """从模板 + session 变量解析完整的 XDF 路径。"""
        from datetime import datetime
        
        path = self.config.path_template
        path = path.replace("%p", participant)
        path = path.replace("%s", session)
        path = path.replace("%c", class_mode)
        path = path.replace("%b", task)
        path = path.replace("%n", str(run))
        # 添加日期占位符：%d = MMDD（例如 0417 表示 4 月 17 日）
        path = path.replace("%d", datetime.now().strftime("%m%d"))
        # 使用字符串拼接以在 Windows 上保留正斜杠，
        # 因为 Path() 会将其转换为反斜杠。
        study_root = self.config.study_root.rstrip("/")
        full_path = f"{study_root}/{path}"
        self._xdf_path = full_path
        return full_path

    def start_recording(self, participant: str, session: str,
                        run: int, task: str, class_mode: str = "binary") -> str:
        """启动 LabRecorderCLI 作为后台进程。

        返回 XDF 文件路径。如果 CLI 可执行文件不存在则抛出
        ``FileNotFoundError``。
        """
        xdf_path = self.resolve_xdf_path(participant, session, run, task, class_mode)

        # 确保输出目录存在
        Path(xdf_path).parent.mkdir(parents=True, exist_ok=True)

        if not self.config.cli_path.exists():
            raise FileNotFoundError(
                f"LabRecorderCLI not found: {self.config.cli_path}"
            )

        cmd: list[str] = [str(self.config.cli_path), xdf_path]
        cmd.extend(self.config.stream_queries)

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        # 启动后台线程捕获 stderr
        self._stderr_lines = []
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True,
            name="LabRecorderCLI-stderr",
        )
        self._stderr_thread.start()
        print(
            f"LabRecorderCLI 已启动 (PID={self.process.pid}), "
            f"XDF: {xdf_path}"
        )
        return xdf_path

    def stop_recording(self) -> None:
        """先优雅停止 LabRecorderCLI，必要时强制终止。"""
        if self.process is None:
            return
        try:
            # 在 Windows 上向进程组发送 CTRL_BREAK_EVENT
            if hasattr(signal, "CTRL_BREAK_EVENT"):
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            if Path(self._xdf_path).exists() and Path(self._xdf_path).stat().st_size > 0:
                print(f"LabRecorderCLI 已停止, XDF 已保存: {self._xdf_path}")
            else:
                print(f"LabRecorderCLI 已停止, 但 XDF 文件为空或不存在: {self._xdf_path}（可能没有 LSL 流可录制）")
        except Exception as exc:
            print(f"WARNING: 停止 LabRecorderCLI 时出错: {exc}")
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            self.process = None

    @property
    def is_recording(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def xdf_path(self) -> str:
        return self._xdf_path

    def check_recording_status(self) -> tuple[bool, str]:
        """检查录制是否活跃，返回状态和错误信息。
        
        返回:
            (is_ok, error_message): is_ok=True 表示录制正常，
            False 表示进程崩溃；error_message 包含崩溃时的 stderr。
        """
        if self.process is None:
            return (False, "LabRecorderCLI was not started")
        
        poll_result = self.process.poll()
        if poll_result is None:
            # 进程仍在运行
            return (True, "")
        
        # 进程已退出 — 捕获 stderr
        stderr_log = self.get_stderr_log()
        error_msg = (
            f"LabRecorderCLI crashed (exit code: {poll_result})\n"
            f"stderr:\n{stderr_log}" if stderr_log else
            f"LabRecorderCLI crashed (exit code: {poll_result})\n"
            f"(no stderr output)"
        )
        return (False, error_msg)


# LSL→CSV 录制器 —— 后台线程持续拉取 LSL 数据写入 CSV
class LslCsvRecorder:
    """从 LSL 流录制原始 EEG + marker 数据到 CSV。

    在后台线程中从已解析的 LSL 流拉取样本，并以兼容旧版录制的格式
    写入 CSV 文件。

    CSV 格式（兼容旧版 OpenBCI 录制）:
      - 第 1 行: 通道索引头（0, 1, 2, ..., N）
      - 后续行: marker_value, ch0, ch1, ..., chN-1

    Marker 来源策略:
      录制器优先使用嵌有 marker 通道的 EEG 流。
      OpenBCI GUI 的 TimeSeriesRaw LSL 流 ("obci_eeg1") 通常将
      marker 作为第一个通道，产生 9 个通道（marker + 8 EEG）。
      这样生成的输出与旧版 CSV 录制完全一致。

      如果 EEG 流的通道数符合预期（含 marker 为 9 通道），则第一个
      通道被视为 marker 列，并按需转换为旧版格式。不需要单独的
      Marker 流。

      如果 EEG 流恰好有 8 个通道（无内嵌 marker），录制器会回退到
      独立的 "Markers" 流 ("obci_eeg2")。
      在此模式下，marker 值在事件之间保持：非零 marker 会持续
      到下一个非零 marker 到达，而 Marker 流的零值 "marker off"
      事件将被忽略。

    Marker 模式:
      - "legacy": 将 marker 映射为 0/1/2（0=其他, 1=左手, 2=右手），
        用于向后兼容旧版处理流程。
      - "detailed": 使用 markers.py 中定义的实际数字 marker 值
        （5, 11, 12, 21, 22 等）。
    """

    # 从详细 marker 值到旧版 0/1/2 的映射
    _LEGACY_MARKER_MAP: dict[int, int] = {}

    @classmethod
    def _build_legacy_map(cls) -> None:
        """从 markers.py 的 MARKERS 字典构建旧版 marker 映射。"""
        if cls._LEGACY_MARKER_MAP:
            return  # 已经构建
        for k in MARKERS:
            if "left" in k.lower():
                cls._LEGACY_MARKER_MAP[MARKERS[k]] = 1
            elif "right" in k.lower():
                cls._LEGACY_MARKER_MAP[MARKERS[k]] = 2

    def __init__(
        self,
        csv_path: str,
        marker_mode: str = "legacy",
        resolve_timeout: float = 10.0,
    ) -> None:
        self._csv_path = csv_path
        self._marker_mode = marker_mode
        self._resolve_timeout = resolve_timeout
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: str = ""
        self._samples_written = 0
        self._file: Any = None
        self._writer: Any = None
        self._eeg_inlet: Any = None
        self._marker_inlet: Any = None
        self._embedded_marker: bool = False  # True 表示 EEG 流包含 marker 通道

    def _convert_marker(self, raw_marker: float) -> int | float:
        """根据配置的 marker 模式转换原始 marker 值。"""
        if self._marker_mode == "legacy":
            int_marker = int(round(raw_marker))
            return self._LEGACY_MARKER_MAP.get(int_marker, 0)
        return raw_marker

    def start_recording(self) -> str:
        """启动 CSV 录制线程。

        解析 LSL 流，打开 CSV 文件，并在后台线程中开始拉取样本。
        返回 CSV 文件路径。

        如果 pylsl 未安装或无法解析流，则抛出 RuntimeError。
        """
        if pylsl is None:
            raise RuntimeError("pylsl 未安装，无法录制 CSV")

        self._build_legacy_map()

        # ── 按名称解析 EEG 流（OpenBCI GUI 默认: "obci_eeg1"）──
        eeg_streams = pylsl.resolve_byprop(
            "name", "obci_eeg1", timeout=self._resolve_timeout, minimum=1,
        )
        if not eeg_streams:
            raise RuntimeError(
                f"无法解析 LSL EEG 流 (name=obci_eeg1, "
                f"timeout={self._resolve_timeout}s). "
                f"请确保 OpenBCI GUI 的 Networking Widget 已开启 LSL 输出。"
            )

        self._eeg_inlet = pylsl.StreamInlet(eeg_streams[0], max_buflen=360)
        n_ch = self._eeg_inlet.info().channel_count()

        # 检测 EEG 流是否包含内嵌 marker 通道。
        # 8 通道板 + Marker Widget 的 OpenBCI GUI → 9 通道
        # (marker + 8 EEG)。没有 Marker Widget → 8 通道。
        #
        # 当通道数恰好比预期 EEG 通道数（8）多一个时，
        # 我们将第一个通道视为 marker。
        eeg_channel_count = 8  # 根据 README 期望的 EEG 通道数
        if n_ch == eeg_channel_count + 1:
            # EEG 流包含一个作为第一个通道的 marker 通道
            self._embedded_marker = True
            print(
                f"LslCsvRecorder: EEG 流含 {n_ch} 通道（含内嵌 marker），"
                f"将使用内嵌 marker 通道"
            )
        else:
            # 无内嵌 marker — 回退到独立 Markers 流
            self._embedded_marker = False
            marker_streams = pylsl.resolve_byprop(
                "name", "obci_eeg2", timeout=self._resolve_timeout, minimum=1,
            )
            self._marker_inlet = (
                pylsl.StreamInlet(marker_streams[0], max_buflen=360)
                if marker_streams else None
            )
            if self._marker_inlet is None:
                print(
                    "LslCsvRecorder: 警告 — 未找到独立 Marker 流，"
                    "CSV 的 marker 列将始终为 0"
                )
            else:
                print(
                    f"LslCsvRecorder: EEG 流含 {n_ch} 通道（无内嵌 marker），"
                    f"使用独立 Marker 流"
                )

        # 打开 CSV 文件并写入头部
        csv_path = Path(self._csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)

        if self._embedded_marker:
            # 有内嵌 marker：直接写入所有通道
            # 第 0 列 = marker，第 1..8 列 = EEG 通道
            self._writer.writerow(list(range(n_ch)))
        else:
            # 无内嵌 marker：写入 marker（来自独立流）+ EEG
            self._writer.writerow(list(range(n_ch + 1)))
        self._file.flush()

        # 启动录制线程
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._recording_loop,
            daemon=True,
            name="LslCsvRecorder",
        )
        self._thread.start()
        mode_str = (
            "embedded_marker" if self._embedded_marker
            else "separate_marker_stream"
        )
        print(
            f"LslCsvRecorder 已启动, CSV: {csv_path} "
            f"(marker_mode={self._marker_mode}, strategy={mode_str}, channels={n_ch})"
        )
        return self._csv_path

    def _recording_loop(self) -> None:
        """后台线程：拉取 LSL 样本并写入 CSV。"""
        # 独立 marker 流：保持的 marker 值
        # 仅在非零 marker 时更新；零值 marker off 事件
        # 被忽略，这样前一个非零 marker 得以保持。
        held_marker = 0.0

        try:
            while not self._stop_event.is_set():
                # ── 从独立流拉取 marker（非内嵌模式）──
                if not self._embedded_marker and self._marker_inlet is not None:
                    try:
                        sample, _ts = self._marker_inlet.pull_sample(timeout=0.0)
                        if sample is not None:
                            new_marker = float(sample[0])
                            # 仅在非零值时更新 held_marker。
                            # OpenBCI GUI 在每次 marker 之后发送一个值为 0 的
                            # marker off 事件 — 我们忽略这些，以使最后一个
                            # 非零 marker 一直保持到下一个到来。
                            if new_marker != 0.0:
                                held_marker = new_marker
                    except Exception:
                        pass

                # ── 拉取 EEG 样本 ──
                try:
                    sample, _ts = self._eeg_inlet.pull_sample(timeout=0.01)
                    if sample is not None:
                        if self._embedded_marker:
                            # 第一个通道是内嵌 marker
                            raw_marker = float(sample[0])
                            eeg_data = sample[1:]
                            csv_marker = self._convert_marker(raw_marker)
                            row = [csv_marker] + [float(v) for v in eeg_data]
                        else:
                            # marker 来自独立流（保持值）
                            csv_marker = self._convert_marker(held_marker)
                            row = [csv_marker] + [float(v) for v in sample]

                        self._writer.writerow(row)
                        self._samples_written += 1
                        if self._samples_written % 256 == 0:
                            self._file.flush()
                except Exception:
                    pass

        except Exception as exc:
            self._error = str(exc)
        finally:
            if self._file is not None:
                try:
                    self._file.flush()
                except Exception:
                    pass

    def stop_recording(self) -> None:
        """停止录制线程并关闭 CSV 文件。"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        # 排空 EEG 缓冲区的剩余样本
        if self._eeg_inlet is not None:
            try:
                while True:
                    sample, _ts = self._eeg_inlet.pull_sample(timeout=0.0)
                    if sample is None:
                        break
                    if self._embedded_marker:
                        raw_marker = float(sample[0])
                        eeg_data = sample[1:]
                        csv_marker = self._convert_marker(raw_marker)
                        row = [csv_marker] + [float(v) for v in eeg_data]
                    else:
                        row = [0] + [float(v) for v in sample]
                    self._writer.writerow(row)
                    self._samples_written += 1
            except Exception:
                pass

        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

        csv_path = Path(self._csv_path)
        if csv_path.exists() and csv_path.stat().st_size > 0:
            print(
                f"LslCsvRecorder 已停止, CSV 已保存: {csv_path} "
                f"({self._samples_written} samples)"
            )
        else:
            print(f"LslCsvRecorder 已停止, 但 CSV 文件为空或不存在: {csv_path}")

    @property
    def is_recording(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def samples_written(self) -> int:
        return self._samples_written

    @property
    def error(self) -> str:
        return self._error

    @property
    def csv_path(self) -> str:
        return self._csv_path


# 实时分类器子进程管理器
class RTClassifierManager:
    """管理 realtime_classifier.py 子进程。"""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        """启动分类器子进程。"""
        rt_cfg = self._config.ssvep_rt
        if not rt_cfg.enabled:
            return

        cmd = [
            sys.executable, "realtime_classifier.py",
            "--timeout", "30",
            "--window-size-s", str(rt_cfg.classifier_window_s),
            "--stride-s", str(rt_cfg.classifier_stride_s),
            "--confidence-threshold", str(rt_cfg.confidence_threshold),
            "--max-windows", "0",       # 无限制 — 让所有窗口投票以获得稳健多数
            "--skip-initial", "0",       # 不跳过任何窗口 — 每个窗口都计入多数投票
        ]
        if rt_cfg.mi_enabled and rt_cfg.mi_checkpoint_path and rt_cfg.mi_checkpoint_path.strip():
            cmd.extend(["--mi-checkpoint", rt_cfg.mi_checkpoint_path])
        else:
            cmd.append("--no-mi")
        # display_mode 为 both_sides 时传递双面模式参数
        if rt_cfg.display_mode == "both_sides":
            cmd.append("--both-sides")
        # LSL diagnostics
        if rt_cfg.enable_diag:
            cmd.extend(["--diag", "--diag-dir", rt_cfg.diag_dir])

        self._process = subprocess.Popen(
            cmd,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        _rt_logger.info("Classifier subprocess started (PID=%s)", self._process.pid)

    def stop(self) -> None:
        """优雅停止分类器子进程。"""
        if self._process is None:
            return
        try:
            self._process.send_signal(
                signal.CTRL_BREAK_EVENT if hasattr(signal, "CTRL_BREAK_EVENT") else signal.SIGTERM
            )
        except Exception:
            pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=3)
        _rt_logger.info("Classifier subprocess stopped.")
        self._process = None

    @property
    def is_running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None


# 分类反馈显示 —— 非阻塞 LSL inlet
class ClassificationFeedbackDisplay:
    """分类反馈显示的非阻塞 LSL inlet。

    设计：在渲染循环之前连接一次，然后在紧密循环中仅做非阻塞拉取。
    绝不要在 PsychoPy 的 draw/flip 路径中调用 try_connect
    （它会做阻塞 LSL resolve_byprop）—— 单次 100ms 的解析
    在 16.67ms 的帧预算内会导致级联丢帧。
    """

    def __init__(self) -> None:
        self._inlet = None
        self._connected = False
        self._last_label: str = ""
        self._last_confidence: float = 0.0

    def try_connect(self, timeout: float = 0.5) -> bool:
        """尝试连接到 classification_result LSL 流。

        在进入渲染循环之前调用一次。不要从 draw/flip 路径中调用。
        """
        try:
            import pylsl as _pylsl
            streams = _pylsl.resolve_byprop(
                "type", "classification_result", timeout=timeout, minimum=1,
            )
            if not streams:
                return False
            self._inlet = _pylsl.StreamInlet(streams[0])
            self._connected = True
            return True
        except Exception:
            return False

    def ensure_connected(self, timeout: float = 0.1) -> None:
        """如果尚未连接，尝试连接一次。

        在相位边界调用是安全的，但仍会执行一次阻塞 LSL
        resolve — 尽可能在进入紧密渲染循环之前显式调用
        try_connect。
        """
        if not self._connected:
            self.try_connect(timeout=timeout)

    def pull_result(self) -> dict | None:
        """非阻塞拉取最新的分类结果。"""
        if not self._inlet:
            return None
        try:
            sample, _ts = self._inlet.pull_sample(timeout=0.0)
            if sample is not None:
                label_int = int(sample[0])
                return {
                    "label": "left" if label_int == 0 else "right",
                    "confidence": float(sample[1]),
                }
            return None
        except Exception:
            return None

    def get_feedback_text(self) -> str:
        """获取当前反馈文本 — 零阻塞，对渲染循环安全。"""
        result = self.pull_result()
        if result is not None:
            self._last_label = result["label"]
            self._last_confidence = result["confidence"]

        if not self._last_label:
            return ""
        if self._last_confidence < 0.05:
            return "检测中..."
        if self._last_label == "left":
            return "检测到：左手意图"
        elif self._last_label == "right":
            return "检测到：右手意图"
        return ""

    def clear(self) -> None:
        """清除粘性反馈标签 — 在 trial 开始时调用以防止显示上一个 trial 的结果。"""
        self._last_label = ""
        self._last_confidence = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected


# 会话 UI —— 管理所有视觉刺激的绘制与缓存
class SessionUI:
    # 缓存策略：image/rect/text stim 缓存避免每帧分配对象
    def __init__(self, win: visual.Window, stimuli: StimulusConfig, display: DisplayConfig) -> None:
        self.win = win
        self.stimuli = stimuli
        self.display = display
        self.title_pos = (0.0, 0.28)
        self.body_pos = (0.0, -0.06)
        self._image_draw_size_cache: dict[tuple[str, float], tuple[float, float]] = {}
        self._image_stim_cache: dict[tuple[str, tuple[float, float], float], visual.ImageStim] = {}
        self._rect_stim_cache: dict[tuple[tuple[float, float], tuple[float, float], float], visual.Rect] = {}
        self._rect_param_cache: dict[tuple[tuple[float, float], tuple[float, float], float], tuple] = {}
        self._text_stim_cache: dict[tuple[tuple[float, float], float, float, bool], visual.TextStim] = {}
        self._text_param_cache: dict[tuple[tuple[float, float], float, float, bool], tuple] = {}
        self._ao_video_pool: dict[str, visual.MovieStim] = {}
        self._ao_video_path: Path | None = None
        self._ao_video_start_s: float = 0.0


    def _apply_info_layout(self) -> None:
        self.title_pos = (0.0, 0.28)
        self.body_pos = (0.0, -0.06)

    def _apply_stimulus_layout(self) -> None:
        self.title_pos = (0.0, 0.34)
        self.body_pos = (0.0, 0.0)

    def _apply_title_only_layout(self) -> None:
        self.title_pos = (0.0, 0.0)
        self.body_pos = (0.0, -0.12)

    def _draw_title(self, title: str) -> None:
        self._draw_cached_text(
            text=title,
            pos=self.title_pos,
            height=0.07,
            wrap_width=1.45,
            color="white",
            bold=True,
        )

    def _draw_body(self, body: str) -> None:
        self._draw_cached_text(
            text=body,
            pos=self.body_pos,
            height=0.042,
            wrap_width=1.3,
            color="white",
        )

    def _draw_phase_image(self, image_path: Path | None, image_scale: float = 1.0) -> None:
        self._draw_positioned_image(
            image_path,
            pos=(0, -0.02),
            target_height=self.stimuli.image_height * image_scale,
        )

    def _get_image_draw_size(self, image_path: Path | None, target_height: float) -> tuple[float, float] | None:
        if image_path is None or not image_path.exists():
            return None

        cache_key = (str(image_path.resolve()), float(target_height))
        cached = self._image_draw_size_cache.get(cache_key)
        if cached is not None:
            return cached

        image = visual.ImageStim(
            self.win,
            image=str(image_path),
            pos=(0, 0),
            units="height",
        )
        original_width, original_height = image.size
        if original_height:
            aspect_ratio = original_width / original_height
        else:
            aspect_ratio = 1.0
        draw_size = (target_height * aspect_ratio, target_height)
        self._image_draw_size_cache[cache_key] = draw_size
        return draw_size

    def _draw_positioned_image(
        self,
        image_path: Path | None,
        *,
        pos: tuple[float, float],
        target_height: float,
        opacity: float = 1.0,
    ) -> tuple[float, float] | None:
        draw_size = self._get_image_draw_size(image_path, target_height)
        if draw_size is None or image_path is None:
            return None

        cache_key = (str(image_path.resolve()), pos, float(target_height))
        image = self._image_stim_cache.get(cache_key)
        if image is None:
            image = visual.ImageStim(
                self.win,
                image=str(image_path),
                pos=pos,
                units="height",
            )
            image.size = draw_size
            self._image_stim_cache[cache_key] = image

        image.pos = pos
        image.size = draw_size
        image.opacity = opacity
        image.draw()
        return draw_size

    # 参数不变时跳过 setter 调用，避免标记 stim dirty
    def _draw_border(
        self,
        *,
        pos: tuple[float, float],
        size: tuple[float, float],
        color: str,
        line_width: float,
    ) -> None:
        cache_key = (pos, size, float(line_width))
        rect = self._rect_stim_cache.get(cache_key)
        if rect is None:
            rect = visual.Rect(
                self.win,
                width=size[0],
                height=size[1],
                pos=pos,
                fillColor=None,
                lineWidth=line_width,
            )
            self._rect_stim_cache[cache_key] = rect
            self._rect_param_cache[cache_key] = (pos, size, color, float(line_width))

        # 参数不变时跳过 setter 调用 — 避免标记 stim dirty
        last = self._rect_param_cache.get(cache_key)
        new_params = (pos, size, color, float(line_width))
        if last is None or last[0] != pos:
            rect.pos = pos
        if last is None or last[1] != size:
            rect.width = size[0]
            rect.height = size[1]
        if last is None or last[2] != color:
            rect.lineColor = color
        if last is None or last[3] != float(line_width):
            rect.lineWidth = line_width
        self._rect_param_cache[cache_key] = new_params
        rect.draw()

    # 缓存 key = (pos, height, wrap_width, bold)
    def _draw_cached_text(
        self,
        *,
        text: str,
        pos: tuple[float, float],
        height: float,
        color: str = "white",
        wrap_width: float | None = None,
        bold: bool = False,
    ) -> None:
        effective_wrap_width = 0.0 if wrap_width is None else wrap_width
        cache_key = (pos, float(height), float(effective_wrap_width), bold)
        stim = self._text_stim_cache.get(cache_key)
        if stim is None:
            stim = visual.TextStim(
                self.win,
                text=text,
                pos=pos,
                height=height,
                wrapWidth=wrap_width,
                color=color,
                bold=bold,
                font="Microsoft YaHei",
            )
            self._text_stim_cache[cache_key] = stim
            self._text_param_cache[cache_key] = (pos, float(height), wrap_width, color, text)

        # 参数不变时跳过 setter 调用 — 避免标记 stim dirty
        last = self._text_param_cache.get(cache_key)
        new_params = (pos, float(height), wrap_width, color, text)
        if last is None or last[:4] != new_params[:4]:
            stim.pos = pos
            stim.height = height
            stim.wrapWidth = wrap_width
            stim.color = color
        if last is None or last[4] != text:
            stim.text = text
        self._text_param_cache[cache_key] = new_params
        stim.draw()

    # display_mode 分发：both_sides / single_center / single_side
    def draw_dual_cue_screen(
        self,
        *,
        title: str,
        target_side: str,
        left_x_pos: float,
        right_x_pos: float,
        y_pos: float,
        image_height: float,
        target_border_color: str,
        nontarget_border_color: str,
        display_mode: str = "both_sides",
    ) -> None:
        self._apply_stimulus_layout()
        self._draw_title(title)

        # 根据 display_mode 决定显示哪些侧
        show_left = True
        show_right = True
        if display_mode in ("single_center", "single_side"):
            show_left = (target_side == "left")
            show_right = (target_side == "right")

        # 根据 display_mode 决定位置
        if display_mode == "single_center":
            left_pos = (0, y_pos) if show_left else (left_x_pos, y_pos)
            right_pos = (0, y_pos) if show_right else (right_x_pos, y_pos)
        else:
            left_pos = (left_x_pos, y_pos)
            right_pos = (right_x_pos, y_pos)

        left_size = None
        right_size = None

        if show_left:
            left_size = self._draw_positioned_image(
                self.stimuli.pure_mi_cue_left_image_path or self.stimuli.cue_image_path,
                pos=left_pos,
                target_height=image_height,
            )
        if show_right:
            right_size = self._draw_positioned_image(
                self.stimuli.pure_mi_cue_right_image_path or self.stimuli.cue_image_path,
                pos=right_pos,
                target_height=image_height,
            )

        left_border_color = target_border_color if target_side == "left" else nontarget_border_color
        right_border_color = target_border_color if target_side == "right" else nontarget_border_color
        if show_left and left_size is not None:
            self._draw_border(
                pos=left_pos,
                size=(left_size[0] + 0.03, left_size[1] + 0.03),
                color=left_border_color,
                line_width=4.0,
            )
        if show_right and right_size is not None:
            self._draw_border(
                pos=right_pos,
                size=(right_size[0] + 0.03, right_size[1] + 0.03),
                color=right_border_color,
                line_width=4.0,
            )

    def draw_text_screen(
        self,
        title: str,
        body: str = "",
        footer: str = "",
        image_path: Path | None = None,
        center_drawer: Callable[[], None] | None = None,
        image_scale: float = 1.0,
        layout: str = "info",
    ) -> None:
        if layout == "stimulus":
            self._apply_stimulus_layout()
        elif layout == "title_only":
            self._apply_title_only_layout()
        else:
            self._apply_info_layout()

        self._draw_title(title)
        if layout == "info" and body:
            self._draw_body(body)
        self._draw_phase_image(image_path, image_scale=image_scale)
        if center_drawer is not None:
            center_drawer()

    def draw_fixation(self) -> None:
        visual.TextStim(
            self.win,
            text="\u25cb",
            height=0.28,
            color="white",
            font="Microsoft YaHei",
        ).draw()

    def draw_cue_cross(self) -> None:
        visual.TextStim(
            self.win,
            text="+",
            height=0.18,
            color="white",
        ).draw()

    def draw_arrow_cue(self, condition: str) -> None:
        self._apply_stimulus_layout()
        self._draw_title(CONDITION_TO_ARROW_CUE_TEXT[condition])
        arrow = "\u2190" if condition == "left" else "\u2192"
        self._draw_cached_text(
            text=arrow,
            pos=(0, 0),
            height=0.35,
            color="white",
        )

    def draw_arrow_mi_task(self, condition: str) -> None:
        self._apply_stimulus_layout()
        self._draw_title(CONDITION_TO_ARROW_MI_TEXT[condition])
        arrow = "\u2190" if condition == "left" else "\u2192"
        self._draw_cached_text(
            text=arrow,
            pos=(0, 0),
            height=0.35,
            color="white",
        )

    def draw_arousal_cue(self, condition: str, cue_style: str) -> None:
        self._apply_stimulus_layout()
        self._draw_title(CONDITION_TO_AROUSAL_CUE_TEXT[condition])

        if cue_style == "arrow":
            arrow = "\u2190" if condition == "left" else "\u2192"
            self._draw_cached_text(
                text=arrow,
                pos=(0, 0),
                height=0.35,
                color="white",
            )
        elif cue_style == "image":
            image_path = (
                self.stimuli.pure_mi_cue_left_image_path
                if condition == "left"
                else self.stimuli.pure_mi_cue_right_image_path
            )
            self._draw_positioned_image(
                image_path,
                pos=(0, 0),
                target_height=self.stimuli.image_height * self.stimuli.task_image_scale,
            )

    # 中央单点 SSVEP 闪烁，不编码左右方向
    def draw_arousal_task_frame(
        self,
        condition: str,
        task_style: str,
        elapsed_time_s: float,
        freq_hz: float,
        waveform: str,
        dim_opacity: float,
    ) -> None:
        self._apply_stimulus_layout()
        self._draw_title(CONDITION_TO_AROUSAL_TASK_TEXT[condition])

        flicker_opacity = self._flicker_opacity(elapsed_time_s, freq_hz, waveform)
        visible_opacity = dim_opacity + flicker_opacity * (1.0 - dim_opacity)

        if task_style == "arrow":
            arrow = "\u2190" if condition == "left" else "\u2192"
            self._draw_cached_text(
                text=arrow,
                pos=(0, 0),
                height=0.35,
                color="white",
            ).opacity = visible_opacity
        elif task_style == "image":
            image_path = (
                self.stimuli.ssvep_left_clean_image_path or self.stimuli.pure_mi_left_image_path
                if condition == "left"
                else self.stimuli.ssvep_right_clean_image_path or self.stimuli.pure_mi_right_image_path
            )
            self._draw_positioned_image(
                image_path,
                pos=(0, 0),
                target_height=self.stimuli.image_height * self.stimuli.task_image_scale,
                opacity=visible_opacity,
            )

    # frequency_coded 模式左右独立频率编码条件 / same_freq 模式不编码
    def draw_serial_ssvep_cue_frame(
        self,
        condition: str,
        ssvep_serial: SSVEPSerialConfig,
        elapsed_time_s: float,
    ) -> None:
        self._apply_stimulus_layout()
        self._draw_title(CONDITION_TO_SERIAL_SSVEP_CUE_TEXT[condition])

        # 根据模式确定频率
        if ssvep_serial.cue_ssvep_mode == "frequency_coded":
            left_freq_hz = ssvep_serial.cue_ssvep_freq_left_hz
            right_freq_hz = ssvep_serial.cue_ssvep_freq_right_hz
        else:  # same_freq
            left_freq_hz = ssvep_serial.same_freq_hz
            right_freq_hz = ssvep_serial.same_freq_hz

        # 根据显示模式和条件决定显示哪些侧
        show_left = True
        show_right = True
        if ssvep_serial.display_mode == "single_center":
            show_left = (condition == "left")
            show_right = (condition == "right")

        left_opacity = self._flicker_opacity(elapsed_time_s, left_freq_hz, ssvep_serial.waveform)
        right_opacity = self._flicker_opacity(elapsed_time_s, right_freq_hz, ssvep_serial.waveform)
        left_vis_opacity = ssvep_serial.dim_opacity + left_opacity * (1.0 - ssvep_serial.dim_opacity)
        right_vis_opacity = ssvep_serial.dim_opacity + right_opacity * (1.0 - ssvep_serial.dim_opacity)

        # 确定位置
        if ssvep_serial.display_mode == "single_center":
            left_pos = (0, 0) if show_left else (-0.38, 0)
            right_pos = (0, 0) if show_right else (0.38, 0)
        else:  # both_sides
            left_pos = (-0.38, 0)
            right_pos = (0.38, 0)

        if ssvep_serial.cue_style == "arrow":
            if show_left:
                self._draw_cached_text(
                    text="\u2190",
                    pos=left_pos,
                    height=ssvep_serial.arrow_height,
                    color=ssvep_serial.arrow_color,
                ).opacity = left_vis_opacity
            if show_right:
                self._draw_cached_text(
                    text="\u2192",
                    pos=right_pos,
                    height=ssvep_serial.arrow_height,
                    color=ssvep_serial.arrow_color,
                ).opacity = right_vis_opacity
        elif ssvep_serial.cue_style == "image":
            left_image = self.stimuli.ssvep_left_clean_image_path or self.stimuli.pure_mi_cue_left_image_path or self.stimuli.cue_image_path
            right_image = self.stimuli.ssvep_right_clean_image_path or self.stimuli.pure_mi_cue_right_image_path or self.stimuli.cue_image_path
            if show_left:
                self._draw_positioned_image(
                    left_image,
                    pos=left_pos,
                    target_height=ssvep_serial.stimulus_height,
                    opacity=left_vis_opacity,
                )
            if show_right:
                self._draw_positioned_image(
                    right_image,
                    pos=right_pos,
                    target_height=ssvep_serial.stimulus_height,
                    opacity=right_vis_opacity,
                )

    def draw_serial_mi_task(
        self,
        condition: str,
        task_style: str,
    ) -> None:
        self._apply_stimulus_layout()
        self._draw_title(CONDITION_TO_SERIAL_MI_TEXT[condition])

        if task_style == "arrow":
            arrow = "\u2190" if condition == "left" else "\u2192"
            self._draw_cached_text(
                text=arrow,
                pos=(0, 0),
                height=0.35,
                color="white",
            )
        elif task_style == "image":
            image_path = (
                self.stimuli.pure_mi_left_image_path
                if condition == "left"
                else self.stimuli.pure_mi_right_image_path
            )
            self._draw_positioned_image(
                image_path,
                pos=(0, 0),
                target_height=self.stimuli.image_height * self.stimuli.task_image_scale,
            )

    # 时间驱动 flicker，两种模式（image opacity / border color），三种 display_mode
    def draw_ssvep_frame(
        self,
        ssvep: SSVEPConfig,
        *,
        elapsed_time_s: float,
        title: str,
        target_side: str,
        target_freq_hz: float,
    ) -> None:
        self._apply_stimulus_layout()
        self._draw_title(title)

        # 根据 display_mode 决定渲染哪些侧
        show_left = True
        show_right = True
        if ssvep.display_mode in ("single_center", "single_side"):
            show_left = (target_side == "left")
            show_right = (target_side == "right")

        # 时间驱动 flicker opacity — 抗掉帧
        left_opacity = self._flicker_opacity(elapsed_time_s, ssvep.left_freq_hz, ssvep.waveform)
        right_opacity = self._flicker_opacity(elapsed_time_s, ssvep.right_freq_hz, ssvep.waveform)

        # 根据 display_mode 确定位置
        if ssvep.display_mode == "single_center":
            left_pos = (0, ssvep.flicker_y_pos) if show_left else (ssvep.left_x_pos, ssvep.flicker_y_pos)
            right_pos = (0, ssvep.flicker_y_pos) if show_right else (ssvep.right_x_pos, ssvep.flicker_y_pos)
        else:
            left_pos = (ssvep.left_x_pos, ssvep.flicker_y_pos)
            right_pos = (ssvep.right_x_pos, ssvep.flicker_y_pos)

        image_height = ssvep.flicker_size[1]

        left_size = None
        right_size = None

        if ssvep.flicker_mode == "image":
            # 图片闪烁：opacity 由波形驱动（方波取 0/1，正弦取 [0,1]）
            if show_left:
                # 将 [0,1] 映射到视觉范围的 [dim_opacity, 1.0]
                left_vis_opacity = ssvep.dim_opacity + left_opacity * (1.0 - ssvep.dim_opacity)
                left_image = self.stimuli.ssvep_left_clean_image_path or self.stimuli.pure_mi_left_image_path
                left_size = self._draw_positioned_image(
                    left_image,
                    pos=left_pos,
                    target_height=image_height,
                    opacity=left_vis_opacity,
                )
            if show_right:
                right_vis_opacity = ssvep.dim_opacity + right_opacity * (1.0 - ssvep.dim_opacity)
                right_image = self.stimuli.ssvep_right_clean_image_path or self.stimuli.pure_mi_right_image_path
                right_size = self._draw_positioned_image(
                    right_image,
                    pos=right_pos,
                    target_height=image_height,
                    opacity=right_vis_opacity,
                )
        else:
            # 边框闪烁模式：静态图片 + 正弦调制的边框颜色
            border_pad = 0.02
            if show_left:
                left_bright_color = ssvep.target_ring_color if target_side == "left" else ssvep.bright_color
                if ssvep.waveform == "sine":
                    left_border_color = self._interpolate_color(ssvep.dark_color, left_bright_color, left_opacity)
                else:
                    left_border_color = left_bright_color if left_opacity >= 0.5 else ssvep.dark_color
                left_size = self._draw_positioned_image(
                    self.stimuli.pure_mi_left_image_path,
                    pos=left_pos,
                    target_height=image_height,
                )
                if left_size is not None:
                    self._draw_border(
                        pos=left_pos,
                        size=(left_size[0] + border_pad, left_size[1] + border_pad),
                        color=left_border_color,
                        line_width=ssvep.flicker_border_width,
                    )
            if show_right:
                right_bright_color = ssvep.target_ring_color if target_side == "right" else ssvep.bright_color
                if ssvep.waveform == "sine":
                    right_border_color = self._interpolate_color(ssvep.dark_color, right_bright_color, right_opacity)
                else:
                    right_border_color = right_bright_color if right_opacity >= 0.5 else ssvep.dark_color
                right_size = self._draw_positioned_image(
                    self.stimuli.pure_mi_right_image_path,
                    pos=right_pos,
                    target_height=image_height,
                )
                if right_size is not None:
                    self._draw_border(
                        pos=right_pos,
                        size=(right_size[0] + border_pad, right_size[1] + border_pad),
                        color=right_border_color,
                        line_width=ssvep.flicker_border_width,
                    )

        # 频率标签 — 仅显示实际显示的侧
        if show_left:
            label_x = 0 if ssvep.display_mode == "single_center" else ssvep.left_x_pos
            self._draw_cached_text(
                text=f"L {ssvep.left_freq_hz:.1f} Hz",
                height=0.04,
                pos=(label_x, ssvep.flicker_y_pos - ssvep.flicker_size[1] / 2 - 0.07),
            )
        if show_right:
            label_x = 0 if ssvep.display_mode == "single_center" else ssvep.right_x_pos
            self._draw_cached_text(
                text=f"R {ssvep.right_freq_hz:.1f} Hz",
                height=0.04,
                pos=(label_x, ssvep.flicker_y_pos - ssvep.flicker_size[1] / 2 - 0.07),
            )

        # 绘制目标环（最外层，叠在闪烁边框/图片之上）— 仅在 image 模式下
        # 在 border 模式下，闪烁边框本身已经为目标侧染成黄色
        if ssvep.flicker_mode == "image" and target_side in {"left", "right"}:
            if ssvep.display_mode == "single_center":
                target_x = 0
            else:
                target_x = ssvep.left_x_pos if target_side == "left" else ssvep.right_x_pos
            target_size = left_size if target_side == "left" else right_size
            if target_size is not None:
                self._draw_border(
                    pos=(target_x, ssvep.flicker_y_pos),
                    size=(target_size[0] + 0.05, target_size[1] + 0.05),
                    color=ssvep.target_ring_color,
                    line_width=ssvep.target_ring_width,
                )
            gaze_text = "看向目标闪烁并保持运动想象" if ssvep.allow_gaze_shift else "注意目标闪烁并保持中央注视"
            self._draw_cached_text(
                text=f"Target: {target_side} ({target_freq_hz:.1f} Hz)\n{gaze_text}",
                height=0.038,
                wrap_width=1.3,
                pos=(0.0, -0.34),
            )

    # 根据 flashing_side 参数决定当前帧哪侧高亮
    def draw_p300_frame(
        self,
        p300: P300Config,
        *,
        flashing_side: str | None,
        title: str,
        target_side: str,
    ) -> None:
        self._apply_stimulus_layout()
        self._draw_title(title)

        left_pos = (p300.left_x_pos, p300.y_pos)
        right_pos = (p300.right_x_pos, p300.y_pos)
        image_height = p300.image_size[1]

        if p300.flash_mode == "image":
            left_opacity = 1.0 if flashing_side == "left" else p300.dim_opacity
            right_opacity = 1.0 if flashing_side == "right" else p300.dim_opacity
            left_image = self.stimuli.ssvep_left_clean_image_path or self.stimuli.pure_mi_left_image_path
            right_image = self.stimuli.ssvep_right_clean_image_path or self.stimuli.pure_mi_right_image_path

            left_size = self._draw_positioned_image(
                left_image,
                pos=left_pos,
                target_height=image_height,
                opacity=left_opacity,
            )
            right_size = self._draw_positioned_image(
                right_image,
                pos=right_pos,
                target_height=image_height,
                opacity=right_opacity,
            )
        else:
            # 边框闪烁模式（默认）：静态图片 + 闪烁边框
            left_border_color = p300.flash_color if flashing_side == "left" else p300.noflash_color
            right_border_color = p300.flash_color if flashing_side == "right" else p300.noflash_color

            left_size = self._draw_positioned_image(
                self.stimuli.pure_mi_left_image_path,
                pos=left_pos,
                target_height=image_height,
            )
            right_size = self._draw_positioned_image(
                self.stimuli.pure_mi_right_image_path,
                pos=right_pos,
                target_height=image_height,
            )

            border_pad = 0.02
            if left_size is not None:
                self._draw_border(
                    pos=left_pos,
                    size=(left_size[0] + border_pad, left_size[1] + border_pad),
                    color=left_border_color,
                    line_width=p300.flash_border_width,
                )
            if right_size is not None:
                self._draw_border(
                    pos=right_pos,
                    size=(right_size[0] + border_pad, right_size[1] + border_pad),
                    color=right_border_color,
                    line_width=p300.flash_border_width,
                )

        # 绘制目标环（最外层，叠在闪烁边框/图片之上）
        if target_side in {"left", "right"}:
            target_x = p300.left_x_pos if target_side == "left" else p300.right_x_pos
            target_size = left_size if target_side == "left" else right_size
            if target_size is not None:
                self._draw_border(
                    pos=(target_x, p300.y_pos),
                    size=(target_size[0] + 0.05, target_size[1] + 0.05),
                    color=p300.target_ring_color,
                    line_width=p300.target_ring_width,
                )
            self._draw_cached_text(
                text=f"Target: {target_side}    SOA {p300.soa_s * 1000:.0f}ms",
                height=0.038,
                wrap_width=1.3,
                pos=(0.0, -0.34),
            )

    # 时间驱动设计 —— 用 elapsed_time_s 而非帧计数，抗掉帧；square 方波 / sine 正弦波
    def _flicker_opacity(self, elapsed_time_s: float, freq_hz: float, waveform: str = "square") -> float:
        """返回 [0, 1] 范围内的 SSVEP 闪烁 opacity 值。

        时间驱动：使用 elapsed_time_s，因此结果对
        丢帧具有鲁棒性，不需要整数 frames_per_cycle。

        waveform:
          - "square": 每个周期的前半段返回 1.0，
            后半段返回 0.0。SSVEP 响应更强但视觉疲劳更大。
          - "sine": 返回 0.5 * (1 + sin(2pi*ft))，平滑的
            正弦调制。更舒适，无谐波。
        """
        period_s = 1.0 / max(freq_hz, 1e-6)
        phase_in_period = elapsed_time_s % period_s
        if waveform == "sine":
            import math
            return 0.5 * (1.0 + math.sin(2.0 * math.pi * freq_hz * elapsed_time_s))
        else:  # square
            return 1.0 if phase_in_period < (period_s / 2.0) else 0.0

    # 在 PsychoPy [-1,1] 色彩空间中做 RGB 线性插值，输出 hex
    @staticmethod
    def _interpolate_color(color_dark: str, color_bright: str, t: float) -> str:
        """在两个十六进制/色彩空间颜色之间做线性插值。

        t 在 [0, 1] 范围内，0 表示暗色，1 表示亮色。返回十六进制字符串。
        """
        from psychopy.colors import Color
        dark = Color(color_dark).rgb  # (r, g, b) each in [-1, 1]
        bright = Color(color_bright).rgb
        rgb = tuple(dark[i] + t * (bright[i] - dark[i]) for i in range(3))
        # 从 PsychoPy [-1,1] 转换为 [0,255]
        rgb255 = tuple(int(round((v + 1.0) * 127.5)) for v in rgb)
        rgb255 = tuple(max(0, min(255, v)) for v in rgb255)
        return f"#{rgb255[0]:02x}{rgb255[1]:02x}{rgb255[2]:02x}"

    # ------------------------------------------------------------------
    # AO 视频预加载（session 级生命周期）
    # ------------------------------------------------------------------

    # 预加载避免每个 trial 创建 MovieStim 阻塞数秒
    def preload_ao_videos(
        self,
        video_path: Path | None,
        video_start_s: float,
        image_scale: float,
    ) -> None:
        """预加载 AO 视频的 MovieStim 对象，用于左右手条件。

        在 session 开始时一次性调用，以便后续的
        show_video_phase 调用可以复用预加载的对象，
        而不是在每个 trial 上创建新的 MovieStim
        （这会在主线程阻塞数秒）。

        视频创建时设置 noAudio=True 和
        autoStart=False 以最小化初始化开销。
        """
        # 先释放任何先前加载的视频。
        self.release_ao_videos()

        self._ao_video_path = video_path
        self._ao_video_start_s = video_start_s

        if video_path is None or not video_path.exists():
            return

        target_height = self.stimuli.image_height * image_scale

        for condition, flip in [("left", False), ("right", True)]:
            try:
                movie = visual.MovieStim(
                    self.win,
                    str(video_path),
                    flipHoriz=flip,
                    pos=(0, -0.02),
                    units="height",
                    volume=0,
                    noAudio=True,
                    autoStart=False,
                )
            except Exception:
                print(
                    f"WARNING: Failed to preload AO video for {condition} hand, "
                    f"will fall back to static image."
                )
                continue

            # 缩放视频以匹配静态图像的相同目标高度。
            native_size = movie.size
            if native_size[1] > 0:
                aspect_ratio = native_size[0] / native_size[1]
            else:
                aspect_ratio = 16 / 9  # 1280x720 视频的回退
            movie.size = (target_height * aspect_ratio, target_height)

            # 立即暂停 — 当相位需要时才 seek + play。
            movie.pause()

            self._ao_video_pool[condition] = movie

        if self._ao_video_pool:
            print(
                f"AO video preloaded: {video_path.name} "
                f"({len(self._ao_video_pool)} variant(s), "
                f"start_s={video_start_s})"
            )

    def get_ao_video(self, flip_horizontal: bool) -> visual.MovieStim | None:
        """返回给定翻转状态的预加载 AO MovieStim。"""
        key = "right" if flip_horizontal else "left"
        return self._ao_video_pool.get(key)

    def release_ao_videos(self) -> None:
        """暂停并释放所有预加载的 AO 视频。

        可安全多次调用；如果池为空则为空操作。

        我们故意没有在这里调用 movie.unload()，因为
        MovieStim.unload() 在 ffpyplayer 解码器线程仍在运行时可能死锁。
        我们只暂停电影并丢弃引用。OpenGL 纹理在窗口
        关闭时（销毁 GL 上下文时）被释放。
        """
        for _key, movie in self._ao_video_pool.items():
            try:
                movie.pause()
            except Exception:
                pass
        self._ao_video_pool.clear()
        self._ao_video_path = None
        self._ao_video_start_s = 0.0


# CLI 参数定义（仅少量快捷入口，完整参数配置在对话框和 YAML 中完成）
def parse_args() -> argparse.Namespace:
    """解析核心 CLI 参数。所有其他设置来自 YAML 配置。"""
    parser = argparse.ArgumentParser(
        description="PsychoPy MI experiment for OpenBCI GUI + UDP marker workflow.",
        epilog="详细参数请编辑 YAML 配置文件（默认 config_default.yaml）。优先级：CLI > 用户 YAML > 内置默认 YAML。",
    )
    parser.add_argument("--participant", default="P001", help="被试编号")
    parser.add_argument("--session", default="S001", help="Session 编号")
    parser.add_argument("--run", type=int, default=1, help="Run 编号")
    parser.add_argument("--mode", choices=["pilot", "main", "custom"], default="pilot", help="pilot=快速验证 / main=正式采集 / custom=自定义")
    parser.add_argument("--class-mode", choices=["binary", "ternary"], default="binary", help="binary=left/right / ternary=left/right/rest")
    parser.add_argument(
        "--trial-mode",
        choices=["pure_mi", "ao_mi", "mi_ssvep", "pure_ssvep", "mi_p300", "mixed", "mi_arrow", "mi_ssvep_arousal", "mi_ssvep_serial", "mi_audio_fb", "mi_ssvep_rt"],
        default="pure_mi",
        help="实验范式：pure_mi / ao_mi / mi_ssvep / mi_p300 / mixed / mi_arrow / mi_ssvep_arousal / mi_ssvep_serial / mi_audio_fb / mi_ssvep_rt",
    )
    parser.add_argument("--blocks", type=int, default=None, help="覆盖 block 数（默认 pilot=2, main=4）")
    parser.add_argument("--repeats-per-class", type=int, default=None, help="每 block 每类 trial 重复次数")
    parser.add_argument("--fullscreen", action="store_true", help="全屏模式")
    parser.add_argument("--display-index", type=int, default=0, help="PsychoPy screen 索引（0=主屏，1=副屏，从0开始）")
    parser.add_argument("--study-root", default=None, help="XDF 数据根目录（覆盖 YAML 配置）")
    parser.add_argument("--recording-format", choices=["xdf", "csv", "both"], default=None, help="录制格式：xdf/csv/both（覆盖 YAML 配置）")
    parser.add_argument("--csv-marker-mode", choices=["legacy", "detailed"], default=None, help="CSV marker 模式：legacy(0/1/2)/detailed(实际值)（覆盖 YAML 配置）")
    parser.add_argument("--refresh-rate", type=float, default=None, help="显示器刷新率 Hz（覆盖 YAML 配置，SSVEP/P300 需要）")
    parser.add_argument("--config", default="config_default.yaml", help="YAML 配置文件路径")
    # SSVEP Arousal 特定参数（覆盖 YAML）
    parser.add_argument("--ssvep-arousal-freq-mode", default=None, help="SSVEP Arousal 频率模式 (fixed/random)")
    parser.add_argument("--ssvep-arousal-fixed-freq-hz", type=float, default=None, help="SSVEP Arousal 固定频率 Hz")
    parser.add_argument("--ssvep-arousal-freq-min-hz", type=float, default=None, help="SSVEP Arousal 最小频率 Hz")
    parser.add_argument("--ssvep-arousal-freq-max-hz", type=float, default=None, help="SSVEP Arousal 最大频率 Hz")
    parser.add_argument("--ssvep-arousal-waveform", default=None, help="SSVEP Arousal 波形 (square/sine)")
    parser.add_argument("--ssvep-arousal-cue-style", default=None, help="SSVEP Arousal 提示样式 (arrow/image)")
    parser.add_argument("--ssvep-arousal-task-style", default=None, help="SSVEP Arousal 任务样式 (arrow/image)")
    parser.add_argument("--ssvep-arousal-stimulus-size", type=float, default=None, help="SSVEP Arousal 刺激尺寸")
    parser.add_argument("--ssvep-arousal-dim-opacity", type=float, default=None, help="SSVEP Arousal 透明度")
    parser.add_argument("--ssvep-arousal-arrow-color", default=None, help="SSVEP Arousal 箭头颜色")
    parser.add_argument("--ssvep-arousal-arrow-height", type=float, default=None, help="SSVEP Arousal 箭头大小")
    # SSVEP Serial 特定参数（覆盖 YAML）
    parser.add_argument("--ssvep-serial-cue-ssvep-freq-left-hz", type=float, default=None, help="SSVEP Serial 提示左频率 Hz")
    parser.add_argument("--ssvep-serial-cue-ssvep-freq-right-hz", type=float, default=None, help="SSVEP Serial 提示右频率 Hz")
    parser.add_argument("--ssvep-serial-cue-ssvep-mode", default=None, help="SSVEP Serial 提示模式 (frequency_coded/same_freq)")
    parser.add_argument("--ssvep-serial-same-freq-hz", type=float, default=None, help="SSVEP Serial 同频模式频率 Hz")
    parser.add_argument("--ssvep-serial-cue-ssvep-duration-s", type=float, default=None, help="SSVEP Serial 提示时长 s")
    parser.add_argument("--ssvep-serial-gap-duration-s", type=float, default=None, help="SSVEP Serial 间隔时长 s")
    parser.add_argument("--ssvep-serial-mi-duration-s", type=float, default=None, help="SSVEP Serial MI时长 s")
    parser.add_argument("--ssvep-serial-waveform", default=None, help="SSVEP Serial 波形 (square/sine)")
    parser.add_argument("--ssvep-serial-cue-style", default=None, help="SSVEP Serial 提示样式 (arrow/image)")
    parser.add_argument("--ssvep-serial-task-style", default=None, help="SSVEP Serial 任务样式 (arrow/image)")
    parser.add_argument("--ssvep-serial-display-mode", default=None, help="SSVEP Serial 显示模式 (single_center/both_sides)")
    parser.add_argument("--ssvep-serial-stimulus-width", type=float, default=None, help="SSVEP Serial 刺激宽度")
    parser.add_argument("--ssvep-serial-stimulus-height", type=float, default=None, help="SSVEP Serial 刺激高度")
    parser.add_argument("--ssvep-serial-border-width", type=float, default=None, help="SSVEP Serial 边框宽度")
    parser.add_argument("--ssvep-serial-dim-opacity", type=float, default=None, help="SSVEP Serial 透明度")
    parser.add_argument("--ssvep-serial-arrow-color", default=None, help="SSVEP Serial 箭头颜色")
    parser.add_argument("--ssvep-serial-arrow-height", type=float, default=None, help="SSVEP Serial 箭头大小")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# YAML 配置加载与合并
# ---------------------------------------------------------------------------

# 将 YAML 的 (section, key) 映射为扁平 key，对接旧版 CLI 参数名
_YAML_TO_FLAT: dict[tuple[str, str], str] = {
    # general
    ("general", "fullscreen"): "fullscreen",
    ("general", "fixation"): "fixation",
    ("general", "cue"): "cue",
    ("general", "iti"): "iti",
    ("general", "refresh_rate"): "refresh_rate",
    ("general", "seed"): "seed",
    ("general", "repeats_per_class"): "repeats_per_class",
    ("general", "window_style"): "window_style",
    ("general", "window_width"): "window_width",
    ("general", "window_height"): "window_height",
    ("general", "udp_ip"): "udp_ip",
    ("general", "udp_port"): "udp_port",
    ("general", "output_dir"): "output_dir",
    ("general", "image_height"): "image_height",
    ("general", "task_image_scale"): "task_image_scale",
    ("general", "labrecorder_cli_path"): "labrecorder_cli_path",
    ("general", "labrecorder_study_root"): "labrecorder_study_root",
    ("general", "labrecorder_path_template"): "labrecorder_path_template",
    ("general", "labrecorder_auto_record"): "labrecorder_auto_record",
    ("general", "labrecorder_stream_queries"): "labrecorder_stream_queries",
    ("general", "labrecorder_recording_format"): "labrecorder_recording_format",
    ("general", "labrecorder_csv_marker_mode"): "labrecorder_csv_marker_mode",
    # pure_mi
    ("pure_mi", "imagery"): "imagery",
    ("pure_mi", "cue_image"): "cue_image",
    ("pure_mi", "cue_left_image"): "cue_left_image",
    ("pure_mi", "cue_right_image"): "cue_right_image",
    ("pure_mi", "left_image"): "left_image",
    ("pure_mi", "right_image"): "right_image",
    ("pure_mi", "rest_image"): "rest_image",
    # ao_mi
    ("ao_mi", "ao_prime"): "ao_prime",
    ("ao_mi", "ao_mi"): "ao_mi",
    ("ao_mi", "mi_only"): "mi_only",
    ("ao_mi", "ao_left_image"): "ao_left_image",
    ("ao_mi", "ao_right_image"): "ao_right_image",
    ("ao_mi", "ao_video"): "ao_video",
    ("ao_mi", "ao_video_start"): "ao_video_start",
    # mi_ssvep
    ("mi_ssvep", "duration"): "ssvep_duration",
    ("mi_ssvep", "left_freq"): "ssvep_left_freq",
    ("mi_ssvep", "right_freq"): "ssvep_right_freq",
    ("mi_ssvep", "flicker_mode"): "ssvep_flicker_mode",
    ("mi_ssvep", "waveform"): "ssvep_waveform",
    ("mi_ssvep", "border_width"): "ssvep_border_width",
    ("mi_ssvep", "dim_opacity"): "ssvep_dim_opacity",
    ("mi_ssvep", "left_clean_image"): "ssvep_left_clean_image",
    ("mi_ssvep", "right_clean_image"): "ssvep_right_clean_image",
    ("mi_ssvep", "allow_gaze_shift"): "ssvep_allow_gaze_shift",
    ("mi_ssvep", "box_width"): "ssvep_box_width",
    ("mi_ssvep", "box_height"): "ssvep_box_height",
    ("mi_ssvep", "y_pos"): "ssvep_y_pos",
    ("mi_ssvep", "left_x"): "ssvep_left_x",
    ("mi_ssvep", "right_x"): "ssvep_right_x",
    ("mi_ssvep", "display_mode"): "ssvep_display_mode",
    # mi_p300
    ("mi_p300", "duration"): "p300_duration",
    ("mi_p300", "soa"): "p300_soa",
    ("mi_p300", "flash_duration"): "p300_flash_duration",
    ("mi_p300", "seed"): "p300_seed",
    ("mi_p300", "target_probability"): "p300_target_probability",
    ("mi_p300", "flicker_mode"): "p300_flicker_mode",
    ("mi_p300", "border_width"): "p300_border_width",
    ("mi_p300", "dim_opacity"): "p300_dim_opacity",
    ("mi_p300", "box_width"): "p300_box_width",
    ("mi_p300", "box_height"): "p300_box_height",
    ("mi_p300", "y_pos"): "p300_y_pos",
    ("mi_p300", "left_x"): "p300_left_x",
    ("mi_p300", "right_x"): "p300_right_x",
    # mi_arrow
    ("mi_arrow", "arrow_style"): "arrow_style",
    ("mi_arrow", "arrow_color"): "arrow_color",
    ("mi_arrow", "arrow_height"): "arrow_height",
    # mi_ssvep_arousal
    ("mi_ssvep_arousal", "duration"): "ssvep_arousal_duration",
    ("mi_ssvep_arousal", "freq_mode"): "ssvep_arousal_freq_mode",
    ("mi_ssvep_arousal", "fixed_freq_hz"): "ssvep_arousal_fixed_freq_hz",
    ("mi_ssvep_arousal", "freq_min_hz"): "ssvep_arousal_freq_min_hz",
    ("mi_ssvep_arousal", "freq_max_hz"): "ssvep_arousal_freq_max_hz",
    ("mi_ssvep_arousal", "waveform"): "ssvep_arousal_waveform",
    ("mi_ssvep_arousal", "cue_style"): "ssvep_arousal_cue_style",
    ("mi_ssvep_arousal", "task_style"): "ssvep_arousal_task_style",
    ("mi_ssvep_arousal", "stimulus_size"): "ssvep_arousal_stimulus_size",
    ("mi_ssvep_arousal", "dim_opacity"): "ssvep_arousal_dim_opacity",
    ("mi_ssvep_arousal", "arrow_color"): "ssvep_arousal_arrow_color",
    ("mi_ssvep_arousal", "arrow_height"): "ssvep_arousal_arrow_height",
    ("mi_ssvep_serial", "cue_ssvep_freq_left_hz"): "ssvep_serial_cue_ssvep_freq_left_hz",
    ("mi_ssvep_serial", "cue_ssvep_freq_right_hz"): "ssvep_serial_cue_ssvep_freq_right_hz",
    ("mi_ssvep_serial", "cue_ssvep_mode"): "ssvep_serial_cue_ssvep_mode",
    ("mi_ssvep_serial", "same_freq_hz"): "ssvep_serial_same_freq_hz",
    ("mi_ssvep_serial", "cue_ssvep_duration_s"): "ssvep_serial_cue_ssvep_duration_s",
    ("mi_ssvep_serial", "gap_duration_s"): "ssvep_serial_gap_duration_s",
    ("mi_ssvep_serial", "mi_duration_s"): "ssvep_serial_mi_duration_s",
    ("mi_ssvep_serial", "waveform"): "ssvep_serial_waveform",
    ("mi_ssvep_serial", "cue_style"): "ssvep_serial_cue_style",
    ("mi_ssvep_serial", "task_style"): "ssvep_serial_task_style",
    ("mi_ssvep_serial", "display_mode"): "ssvep_serial_display_mode",
    ("mi_ssvep_serial", "stimulus_width"): "ssvep_serial_stimulus_width",
    ("mi_ssvep_serial", "stimulus_height"): "ssvep_serial_stimulus_height",
    ("mi_ssvep_serial", "border_width"): "ssvep_serial_border_width",
    ("mi_ssvep_serial", "dim_opacity"): "ssvep_serial_dim_opacity",
    ("mi_ssvep_serial", "arrow_color"): "ssvep_serial_arrow_color",
    ("mi_ssvep_serial", "arrow_height"): "ssvep_serial_arrow_height",
    # mi_ssvep_rt
    ("mi_ssvep_rt", "mi_enabled"): "ssvep_rt_mi_enabled",
    ("mi_ssvep_rt", "mi_checkpoint_path"): "ssvep_rt_mi_checkpoint_path",
    ("mi_ssvep_rt", "classifier_window_s"): "ssvep_rt_classifier_window_s",
    ("mi_ssvep_rt", "classifier_stride_s"): "ssvep_rt_classifier_stride_s",
    ("mi_ssvep_rt", "confidence_threshold"): "ssvep_rt_confidence_threshold",
    ("mi_ssvep_rt", "left_freq_hz"): "ssvep_rt_left_freq_hz",
    ("mi_ssvep_rt", "right_freq_hz"): "ssvep_rt_right_freq_hz",
    ("mi_ssvep_rt", "flicker_duration_s"): "ssvep_rt_flicker_duration_s",
    ("mi_ssvep_rt", "flicker_mode"): "ssvep_rt_flicker_mode",
    ("mi_ssvep_rt", "display_mode"): "ssvep_rt_display_mode",
    ("mi_ssvep_rt", "waveform"): "ssvep_rt_waveform",
    ("mi_ssvep_rt", "flicker_size"): "ssvep_rt_flicker_size",
    ("mi_ssvep_rt", "flicker_y_pos"): "ssvep_rt_flicker_y_pos",
    ("mi_ssvep_rt", "left_x_pos"): "ssvep_rt_left_x_pos",
    ("mi_ssvep_rt", "right_x_pos"): "ssvep_rt_right_x_pos",
    ("mi_ssvep_rt", "bright_color"): "ssvep_rt_bright_color",
    ("mi_ssvep_rt", "dark_color"): "ssvep_rt_dark_color",
    ("mi_ssvep_rt", "flicker_border_width"): "ssvep_rt_flicker_border_width",
    ("mi_ssvep_rt", "dim_opacity"): "ssvep_rt_dim_opacity",
}

# 反向映射：flat_key -> (yaml_section, yaml_key)
_FLAT_TO_YAML: dict[str, tuple[str, str]] = {v: k for k, v in _YAML_TO_FLAT.items()}


def load_yaml_config(path: Path) -> dict[str, Any]:
    """加载 YAML 配置文件并返回嵌套字典。失败时返回空字典。"""
    if not path.exists():
        print(f"WARNING: config file not found: {path}, using built-in defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _flatten_yaml(yaml_dict: dict[str, Any]) -> dict[str, Any]:
    """使用 _YAML_TO_FLAT 映射将嵌套 YAML section 展平成扁平字典。"""
    flat: dict[str, Any] = {}
    for (section, key), flat_key in _YAML_TO_FLAT.items():
        section_dict = yaml_dict.get(section, {})
        if section_dict and key in section_dict:
            flat[flat_key] = section_dict[key]
    return flat


# 优先级链：对话框 > CLI 显式传入 > YAML 配置 > 内置默认值（对话框是主入口，CLI 仅作快捷覆盖）
def resolve_config(cli_args: argparse.Namespace, yaml_dict: dict[str, Any]) -> dict[str, Any]:
    """合并 YAML 配置与 CLI/对话框覆盖。优先级：对话框 > CLI > YAML > 内置默认值（对话框值在调用前已注入 cli_args）。

    返回一个扁平字典，键名与旧版 CLI 参数名相同，以便 build_config()
    能以最小改动消费它。
    """
    # 1. 将 YAML 展平为扁平字典
    merged = _flatten_yaml(yaml_dict)

    # 2. 直接映射到扁平键的 CLI 参数（仅在用户显式提供时才覆盖；
    #    我们通过与解析器默认值对比来检测）
    cli_overrides: dict[str, Any] = {
        "participant": cli_args.participant,
        "session": cli_args.session,
        "run": cli_args.run,
        "mode": cli_args.mode,
        "class_mode": cli_args.class_mode,
        "trial_mode": cli_args.trial_mode,
        "fullscreen": cli_args.fullscreen,
        "display_index": cli_args.display_index,
    }
    if cli_args.blocks is not None:
        cli_overrides["blocks"] = cli_args.blocks
    if getattr(cli_args, "repeats_per_class", None) is not None:
        cli_overrides["repeats_per_class"] = cli_args.repeats_per_class
    if getattr(cli_args, "study_root", None) is not None:
        cli_overrides["labrecorder_study_root"] = cli_args.study_root
    if getattr(cli_args, "recording_format", None) is not None:
        cli_overrides["labrecorder_recording_format"] = cli_args.recording_format
    if getattr(cli_args, "csv_marker_mode", None) is not None:
        cli_overrides["labrecorder_csv_marker_mode"] = cli_args.csv_marker_mode
    if getattr(cli_args, "refresh_rate", None) is not None:
        cli_overrides["refresh_rate"] = cli_args.refresh_rate
    # 来自会话对话框的 SSVEP 特定覆盖
    if getattr(cli_args, "ssvep_flicker_mode", None) is not None:
        cli_overrides["ssvep_flicker_mode"] = cli_args.ssvep_flicker_mode
    if getattr(cli_args, "ssvep_waveform", None) is not None:
        cli_overrides["ssvep_waveform"] = cli_args.ssvep_waveform
    if getattr(cli_args, "ssvep_display_mode", None) is not None:
        cli_overrides["ssvep_display_mode"] = cli_args.ssvep_display_mode
    if getattr(cli_args, "ssvep_left_freq", None) is not None:
        cli_overrides["ssvep_left_freq"] = cli_args.ssvep_left_freq
    if getattr(cli_args, "ssvep_right_freq", None) is not None:
        cli_overrides["ssvep_right_freq"] = cli_args.ssvep_right_freq
    # 来自会话对话框的 P300 特定覆盖
    if getattr(cli_args, "p300_flicker_mode", None) is not None:
        cli_overrides["p300_flicker_mode"] = cli_args.p300_flicker_mode
    if getattr(cli_args, "p300_target_probability", None) is not None:
        cli_overrides["p300_target_probability"] = cli_args.p300_target_probability
    # 来自会话对话框或 CLI 的 SSVEP Arousal 特定覆盖
    if getattr(cli_args, "ssvep_arousal_freq_mode", None) is not None:
        cli_overrides["ssvep_arousal_freq_mode"] = cli_args.ssvep_arousal_freq_mode
    if getattr(cli_args, "ssvep_arousal_fixed_freq_hz", None) is not None:
        cli_overrides["ssvep_arousal_fixed_freq_hz"] = cli_args.ssvep_arousal_fixed_freq_hz
    if getattr(cli_args, "ssvep_arousal_freq_min_hz", None) is not None:
        cli_overrides["ssvep_arousal_freq_min_hz"] = cli_args.ssvep_arousal_freq_min_hz
    if getattr(cli_args, "ssvep_arousal_freq_max_hz", None) is not None:
        cli_overrides["ssvep_arousal_freq_max_hz"] = cli_args.ssvep_arousal_freq_max_hz
    if getattr(cli_args, "ssvep_arousal_waveform", None) is not None:
        cli_overrides["ssvep_arousal_waveform"] = cli_args.ssvep_arousal_waveform
    if getattr(cli_args, "ssvep_arousal_cue_style", None) is not None:
        cli_overrides["ssvep_arousal_cue_style"] = cli_args.ssvep_arousal_cue_style
    if getattr(cli_args, "ssvep_arousal_task_style", None) is not None:
        cli_overrides["ssvep_arousal_task_style"] = cli_args.ssvep_arousal_task_style
    if getattr(cli_args, "ssvep_arousal_stimulus_size", None) is not None:
        cli_overrides["ssvep_arousal_stimulus_size"] = cli_args.ssvep_arousal_stimulus_size
    if getattr(cli_args, "ssvep_arousal_dim_opacity", None) is not None:
        cli_overrides["ssvep_arousal_dim_opacity"] = cli_args.ssvep_arousal_dim_opacity
    if getattr(cli_args, "ssvep_arousal_arrow_color", None) is not None:
        cli_overrides["ssvep_arousal_arrow_color"] = cli_args.ssvep_arousal_arrow_color
    if getattr(cli_args, "ssvep_arousal_arrow_height", None) is not None:
        cli_overrides["ssvep_arousal_arrow_height"] = cli_args.ssvep_arousal_arrow_height
    # 来自会话对话框或 CLI 的 SSVEP Serial 特定覆盖
    if getattr(cli_args, "ssvep_serial_cue_ssvep_freq_left_hz", None) is not None:
        cli_overrides["ssvep_serial_cue_ssvep_freq_left_hz"] = cli_args.ssvep_serial_cue_ssvep_freq_left_hz
    if getattr(cli_args, "ssvep_serial_cue_ssvep_freq_right_hz", None) is not None:
        cli_overrides["ssvep_serial_cue_ssvep_freq_right_hz"] = cli_args.ssvep_serial_cue_ssvep_freq_right_hz
    if getattr(cli_args, "ssvep_serial_cue_ssvep_mode", None) is not None:
        cli_overrides["ssvep_serial_cue_ssvep_mode"] = cli_args.ssvep_serial_cue_ssvep_mode
    if getattr(cli_args, "ssvep_serial_same_freq_hz", None) is not None:
        cli_overrides["ssvep_serial_same_freq_hz"] = cli_args.ssvep_serial_same_freq_hz
    if getattr(cli_args, "ssvep_serial_cue_ssvep_duration_s", None) is not None:
        cli_overrides["ssvep_serial_cue_ssvep_duration_s"] = cli_args.ssvep_serial_cue_ssvep_duration_s
    if getattr(cli_args, "ssvep_serial_gap_duration_s", None) is not None:
        cli_overrides["ssvep_serial_gap_duration_s"] = cli_args.ssvep_serial_gap_duration_s
    if getattr(cli_args, "ssvep_serial_mi_duration_s", None) is not None:
        cli_overrides["ssvep_serial_mi_duration_s"] = cli_args.ssvep_serial_mi_duration_s
    if getattr(cli_args, "ssvep_serial_waveform", None) is not None:
        cli_overrides["ssvep_serial_waveform"] = cli_args.ssvep_serial_waveform
    if getattr(cli_args, "ssvep_serial_cue_style", None) is not None:
        cli_overrides["ssvep_serial_cue_style"] = cli_args.ssvep_serial_cue_style
    if getattr(cli_args, "ssvep_serial_task_style", None) is not None:
        cli_overrides["ssvep_serial_task_style"] = cli_args.ssvep_serial_task_style
    if getattr(cli_args, "ssvep_serial_display_mode", None) is not None:
        cli_overrides["ssvep_serial_display_mode"] = cli_args.ssvep_serial_display_mode
    if getattr(cli_args, "ssvep_serial_stimulus_width", None) is not None:
        cli_overrides["ssvep_serial_stimulus_width"] = cli_args.ssvep_serial_stimulus_width
    if getattr(cli_args, "ssvep_serial_stimulus_height", None) is not None:
        cli_overrides["ssvep_serial_stimulus_height"] = cli_args.ssvep_serial_stimulus_height
    if getattr(cli_args, "ssvep_serial_border_width", None) is not None:
        cli_overrides["ssvep_serial_border_width"] = cli_args.ssvep_serial_border_width
    if getattr(cli_args, "ssvep_serial_dim_opacity", None) is not None:
        cli_overrides["ssvep_serial_dim_opacity"] = cli_args.ssvep_serial_dim_opacity
    if getattr(cli_args, "ssvep_serial_arrow_color", None) is not None:
        cli_overrides["ssvep_serial_arrow_color"] = cli_args.ssvep_serial_arrow_color
    if getattr(cli_args, "ssvep_serial_arrow_height", None) is not None:
        cli_overrides["ssvep_serial_arrow_height"] = cli_args.ssvep_serial_arrow_height
    # 来自会话对话框或 CLI 的 SSVEP RT 特定覆盖
    if getattr(cli_args, "ssvep_rt_mi_enabled", None) is not None:
        cli_overrides["ssvep_rt_mi_enabled"] = cli_args.ssvep_rt_mi_enabled
    if getattr(cli_args, "ssvep_rt_mi_checkpoint_path", None) is not None:
        cli_overrides["ssvep_rt_mi_checkpoint_path"] = cli_args.ssvep_rt_mi_checkpoint_path
    if getattr(cli_args, "ssvep_rt_classifier_window_s", None) is not None:
        cli_overrides["ssvep_rt_classifier_window_s"] = cli_args.ssvep_rt_classifier_window_s
    if getattr(cli_args, "ssvep_rt_classifier_stride_s", None) is not None:
        cli_overrides["ssvep_rt_classifier_stride_s"] = cli_args.ssvep_rt_classifier_stride_s
    if getattr(cli_args, "ssvep_rt_confidence_threshold", None) is not None:
        cli_overrides["ssvep_rt_confidence_threshold"] = cli_args.ssvep_rt_confidence_threshold
    if getattr(cli_args, "ssvep_rt_left_freq_hz", None) is not None:
        cli_overrides["ssvep_rt_left_freq_hz"] = cli_args.ssvep_rt_left_freq_hz
    if getattr(cli_args, "ssvep_rt_right_freq_hz", None) is not None:
        cli_overrides["ssvep_rt_right_freq_hz"] = cli_args.ssvep_rt_right_freq_hz
    if getattr(cli_args, "ssvep_rt_flicker_mode", None) is not None:
        cli_overrides["ssvep_rt_flicker_mode"] = cli_args.ssvep_rt_flicker_mode
    if getattr(cli_args, "ssvep_rt_waveform", None) is not None:
        cli_overrides["ssvep_rt_waveform"] = cli_args.ssvep_rt_waveform
    if getattr(cli_args, "ssvep_rt_display_mode", None) is not None:
        cli_overrides["ssvep_rt_display_mode"] = cli_args.ssvep_rt_display_mode
    if getattr(cli_args, "ssvep_rt_enable_diag", None) is not None:
        cli_overrides["ssvep_rt_enable_diag"] = cli_args.ssvep_rt_enable_diag

    merged.update(cli_overrides)
    return merged


def resolve_optional_path(raw_value: str) -> Path | None:
    value = raw_value.strip()
    return Path(value) if value else None


def get_active_conditions(session_cfg: SessionConfig) -> tuple[str, ...]:
    return CLASS_MODE_TO_CONDITIONS[session_cfg.class_mode]


def get_active_trial_types(session_cfg: SessionConfig) -> tuple[str, ...]:
    return TRIAL_MODE_TO_TYPES[session_cfg.trial_mode]


def get_trial_type_label(trial_type: str) -> str:
    return TRIAL_TYPE_TO_LABEL[trial_type]


# 各范式 trial 的阶段序列定义
def get_phase_sequence_for_trial_type(trial_type: str) -> tuple[str, ...]:
    if trial_type == "pure_mi":
        return ("fixation", "cue", "pure_mi", "iti")
    if trial_type == "mi_ssvep":
        return ("fixation", "cue", "mi_ssvep", "iti")
    if trial_type == "pure_ssvep":
        return ("fixation", "cue", "pure_ssvep", "iti")
    if trial_type == "mi_p300":
        return ("fixation", "cue", "mi_p300", "iti")
    if trial_type == "mi_arrow":
        return ("fixation", "arrow_cue", "arrow_mi", "iti")
    if trial_type == "mi_ssvep_arousal":
        return ("fixation", "arousal_cue", "arousal_task", "iti")
    if trial_type == "mi_ssvep_serial":
        return ("fixation", "serial_ssvep_cue", "serial_gap", "serial_mi", "iti")
    if trial_type == "mi_ssvep_rt":
        return ("fixation", "cue", "mi_ssvep_rt", "iti")
    return ("fixation", "cue", "ao_prime", "ao_mi", "mi_only", "iti")


def get_pure_mi_cue_image_path(stimuli: StimulusConfig, condition: str) -> Path | None:
    if condition == "left":
        return stimuli.pure_mi_cue_left_image_path or stimuli.cue_image_path
    if condition == "right":
        return stimuli.pure_mi_cue_right_image_path or stimuli.cue_image_path
    return stimuli.cue_image_path


def get_pure_mi_image_path(stimuli: StimulusConfig, condition: str) -> Path | None:
    if condition == "left":
        return stimuli.pure_mi_left_image_path
    if condition == "right":
        return stimuli.pure_mi_right_image_path
    return stimuli.pure_mi_rest_image_path


def get_ao_image_path(stimuli: StimulusConfig, condition: str) -> Path | None:
    if condition == "left":
        return stimuli.ao_left_image_path
    if condition == "right":
        return stimuli.ao_right_image_path
    return None



# 递归合并字典：override 值完全覆盖 base 中的同名字典
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归将 override 合并到 base 中。override 的值优先。"""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# 以 OLM/ 项目根目录为基准解析相对路径
def _resolve_project_relative_path(path_str: str) -> Path:
    """解析相对于项目根目录 (OLM/) 的路径。

    如果 path_str 已经是绝对路径，则原样返回。
    否则，相对于脚本目录的父目录（即 OLM/）解析，
    以便 LabRecorder/LabRecorderCLI.exe 无论当前工作目录
    如何都能正确解析。
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    # __file__ 是 mi_psychopy/run_mi_experiment.py → 父目录是 mi_psychopy/ → 再父目录是 OLM/
    project_root = Path(__file__).resolve().parent.parent
    return (project_root / p).resolve()


# 配置装配流水线：YAML 加载 → 扁平化 → 对话框覆盖 → CLI 覆盖 → 校验 → 冻结 dataclass（对话框是参数主入口）
def build_config(args: argparse.Namespace) -> ExperimentConfig:
    # 始终先加载内置默认值作为基础，再叠加用户配置
    script_dir = Path(__file__).resolve().parent
    default_yaml_path = script_dir / "config_default.yaml"
    base_dict = load_yaml_config(default_yaml_path)
    user_yaml_path = Path(args.config)
    if user_yaml_path.resolve() != default_yaml_path.resolve():
        user_dict = load_yaml_config(user_yaml_path)
        if user_dict:
            base_dict = _deep_merge(base_dict, user_dict)
    c = resolve_config(args, base_dict)

    # --- 校验 ---
    if c["trial_mode"] in ("ao_mi", "mixed", "mi_ssvep", "pure_ssvep", "mi_p300", "mi_arrow", "mi_ssvep_arousal", "mi_ssvep_serial", "mi_ssvep_rt") and c["class_mode"] != "binary":
        raise SystemExit("AO+MI / MI+SSVEP / MI+P300 / MI-Arrow / SSVEP Arousal / Serial SSVEP→MI 首版仅支持 binary 模式。请使用 --class-mode binary。")
    if c["refresh_rate"] <= 0:
        raise SystemExit("general.refresh_rate 必须为正数。")
    if c.get("labrecorder_recording_format", "xdf") not in ("xdf", "csv", "both"):
        raise SystemExit("general.labrecorder_recording_format 必须为 xdf / csv / both。")
    if c.get("labrecorder_csv_marker_mode", "legacy") not in ("legacy", "detailed"):
        raise SystemExit("general.labrecorder_csv_marker_mode 必须为 legacy / detailed。")
    if c["ssvep_left_freq"] <= 0 or c["ssvep_right_freq"] <= 0:
        raise SystemExit("mi_ssvep.left_freq / right_freq 必须为正数。")
    if c["ssvep_border_width"] <= 0:
        raise SystemExit("mi_ssvep.border_width 必须为正数。")
    if c["ssvep_flicker_mode"] not in ("border", "image"):
        raise SystemExit("mi_ssvep.flicker_mode 必须为 border 或 image。")
    if c.get("ssvep_waveform", "square") not in ("square", "sine"):
        raise SystemExit("mi_ssvep.waveform 必须为 square 或 sine。")
    if not (0.0 <= c["ssvep_dim_opacity"] <= 1.0):
        raise SystemExit("mi_ssvep.dim_opacity 必须在 [0.0, 1.0] 范围内。")
    if c.get("ssvep_display_mode", "single_side") not in ("both_sides", "single_center", "single_side"):
        raise SystemExit("mi_ssvep.display_mode 必须为 both_sides / single_center / single_side")
    if c["p300_soa"] <= 0:
        raise SystemExit("mi_p300.soa 必须为正数。")
    if c["p300_flash_duration"] <= 0:
        raise SystemExit("mi_p300.flash_duration 必须为正数。")
    if c["p300_flash_duration"] > c["p300_soa"]:
        raise SystemExit("mi_p300.flash_duration 不能超过 mi_p300.soa。")
    if not (0.0 < c["p300_target_probability"] < 1.0):
        raise SystemExit("mi_p300.target_probability 必须在 (0.0, 1.0) 范围内。")
    
    # 验证并自动调整 P300 目标数量
    p300_n_flashes = max(1, int(c["p300_duration"] / c["p300_soa"]))
    p300_n_target = round(p300_n_flashes * c["p300_target_probability"])
    p300_actual_prob = p300_n_target / p300_n_flashes if p300_n_flashes > 0 else 0
    
    if p300_n_target < 1:
        # 自动调整 duration 以确保至少 1 个目标
        min_n_flashes = int(1.0 / c["p300_target_probability"]) + 1
        min_duration = min_n_flashes * c["p300_soa"]
        print(
            f"WARNING: mi_p300 duration={c['p300_duration']}s too short for "
            f"target_probability={c['p300_target_probability']:.2%} (0 targets). "
            f"Auto-adjusting duration to {min_duration:.1f}s for at least 1 target."
        )
        c["p300_duration"] = min_duration
        p300_n_flashes = max(1, int(c["p300_duration"] / c["p300_soa"]))
        p300_n_target = round(p300_n_flashes * c["p300_target_probability"])
        p300_actual_prob = p300_n_target / p300_n_flashes
    
    if abs(p300_actual_prob - c["p300_target_probability"]) > 0.05:
        print(
            f"WARNING: mi_p300 target_probability={c['p300_target_probability']:.2%} "
            f"but actual will be {p300_actual_prob:.2%} ({p300_n_target}/{p300_n_flashes} targets)."
        )
    if c["p300_border_width"] <= 0:
        raise SystemExit("mi_p300.border_width 必须为正数。")
    if c["p300_flicker_mode"] not in ("border", "image"):
        raise SystemExit("mi_p300.flicker_mode 必须为 border 或 image。")
    if not (0.0 <= c["p300_dim_opacity"] <= 1.0):
        raise SystemExit("mi_p300.dim_opacity 必须在 [0.0, 1.0] 范围内。")

    # 注意：使用时间驱动正弦 flicker 时，非整数 frames_per_cycle
    # 不再是正确性问题（sin() 与帧率无关）。
    # 我们保留此信息性注释用于低帧率场景。
    for label, freq in [("left", c["ssvep_left_freq"]), ("right", c["ssvep_right_freq"])]:
        frames_per_cycle = c["refresh_rate"] / freq
        if frames_per_cycle != round(frames_per_cycle):
            print(
                f"INFO: {label} SSVEP freq={freq} Hz does not evenly divide "
                f"refresh_rate={c['refresh_rate']} Hz "
                f"(frames_per_cycle={frames_per_cycle:.2f}). "
                f"This is fine with sinusoidal flicker; would only matter for "
                f"square-wave duty-cycle precision at low frame rates."
            )

    default_blocks = 2 if c["mode"] == "pilot" else 4
    cue_image_path = resolve_optional_path(c["cue_image"])
    pure_mi_cue_left_image_path = resolve_optional_path(c["cue_left_image"]) or cue_image_path
    pure_mi_cue_right_image_path = resolve_optional_path(c["cue_right_image"]) or cue_image_path
    pure_mi_left_image_path = resolve_optional_path(c["left_image"])
    pure_mi_right_image_path = resolve_optional_path(c["right_image"])
    pure_mi_rest_image_path = resolve_optional_path(c["rest_image"]) or cue_image_path
    ao_left_image_path = resolve_optional_path(c["ao_left_image"]) or pure_mi_left_image_path
    ao_right_image_path = resolve_optional_path(c["ao_right_image"]) or pure_mi_right_image_path
    ao_video_path = resolve_optional_path(c["ao_video"])
    ao_video_start_s = float(c.get("ao_video_start", 2.0))

    return ExperimentConfig(
        participant=c["participant"],
        session=c["session"],
        run=c["run"],
        timings=TimingConfig(
            fixation_s=c["fixation"],
            cue_s=c["cue"],
            imagery_s=c["imagery"],
            ao_prime_s=c["ao_prime"],
            ao_mi_s=c["ao_mi"],
            mi_only_s=c["mi_only"],
            iti_s=c["iti"],
        ),
        display=DisplayConfig(
            fullscreen=c["fullscreen"],
            window_width=c["window_width"],
            window_height=c["window_height"],
            window_style=c["window_style"],
            display_index=max(0, c["display_index"]),
            refresh_rate_hz=c["refresh_rate"],
        ),
        stimuli=StimulusConfig(
            cue_image_path=cue_image_path,
            pure_mi_cue_left_image_path=pure_mi_cue_left_image_path,
            pure_mi_cue_right_image_path=pure_mi_cue_right_image_path,
            pure_mi_left_image_path=pure_mi_left_image_path,
            pure_mi_right_image_path=pure_mi_right_image_path,
            pure_mi_rest_image_path=pure_mi_rest_image_path,
            ao_left_image_path=ao_left_image_path,
            ao_right_image_path=ao_right_image_path,
            ao_video_path=ao_video_path,
            ao_video_start_s=ao_video_start_s,
            ssvep_left_clean_image_path=resolve_optional_path(c["ssvep_left_clean_image"]),
            ssvep_right_clean_image_path=resolve_optional_path(c["ssvep_right_clean_image"]),
            image_height=c["image_height"],
            task_image_scale=c["task_image_scale"],
        ),
        ssvep=SSVEPConfig(
            enabled=c["trial_mode"] in ("mi_ssvep", "pure_ssvep"),
            left_freq_hz=c["ssvep_left_freq"],
            right_freq_hz=c["ssvep_right_freq"],
            flicker_duration_s=c["ssvep_duration"],
            allow_gaze_shift=c["ssvep_allow_gaze_shift"],
            flicker_size=(c["ssvep_box_width"], c["ssvep_box_height"]),
            flicker_y_pos=c["ssvep_y_pos"],
            left_x_pos=c["ssvep_left_x"],
            right_x_pos=c["ssvep_right_x"],
            flicker_mode=c["ssvep_flicker_mode"],
            waveform=c.get("ssvep_waveform", "square"),
            display_mode=c.get("ssvep_display_mode", "single_side"),
            flicker_border_width=c["ssvep_border_width"],
            dim_opacity=c["ssvep_dim_opacity"],
        ),
        p300=P300Config(
            enabled=c["trial_mode"] == "mi_p300",
            task_duration_s=c["p300_duration"],
            soa_s=c["p300_soa"],
            flash_duration_s=c["p300_flash_duration"],
            flash_sequence_seed=c["p300_seed"],
            left_x_pos=c["p300_left_x"],
            right_x_pos=c["p300_right_x"],
            y_pos=c["p300_y_pos"],
            image_size=(c["p300_box_width"], c["p300_box_height"]),
            target_probability=c["p300_target_probability"],
            flash_mode=c["p300_flicker_mode"],
            flash_border_width=c["p300_border_width"],
            dim_opacity=c["p300_dim_opacity"],
        ),
        arrow=ArrowConfig(
            enabled=c["trial_mode"] == "mi_arrow",
            arrow_style=c.get("arrow_style", "unicode"),
            arrow_color=c.get("arrow_color", "white"),
            arrow_height=c.get("arrow_height", 0.20),
        ),
        ssvep_arousal=SSVEPArousalConfig(
            enabled=c["trial_mode"] == "mi_ssvep_arousal",
            freq_mode=c.get("ssvep_arousal_freq_mode", "fixed"),
            fixed_freq_hz=c.get("ssvep_arousal_fixed_freq_hz", 20.0),
            freq_min_hz=c.get("ssvep_arousal_freq_min_hz", 18.0),
            freq_max_hz=c.get("ssvep_arousal_freq_max_hz", 25.0),
            waveform=c.get("ssvep_arousal_waveform", "sine"),
            cue_style=c.get("ssvep_arousal_cue_style", "arrow"),
            task_style=c.get("ssvep_arousal_task_style", "arrow"),
            flicker_duration_s=c.get("ssvep_arousal_duration", 4.5),
            stimulus_size=(c.get("ssvep_arousal_stimulus_size", 0.35),
                        c.get("ssvep_arousal_stimulus_size", 0.35)),
            dim_opacity=c.get("ssvep_arousal_dim_opacity", 0.0),
            arrow_color=c.get("ssvep_arousal_arrow_color", "white"),
            arrow_height=c.get("ssvep_arousal_arrow_height", 0.20),
        ),
        ssvep_serial=SSVEPSerialConfig(
            enabled=c["trial_mode"] == "mi_ssvep_serial",
            cue_ssvep_freq_left_hz=c.get("ssvep_serial_cue_ssvep_freq_left_hz", 10.0),
            cue_ssvep_freq_right_hz=c.get("ssvep_serial_cue_ssvep_freq_right_hz", 15.0),
            cue_ssvep_mode=c.get("ssvep_serial_cue_ssvep_mode", "frequency_coded"),
            same_freq_hz=c.get("ssvep_serial_same_freq_hz", 20.0),
            cue_ssvep_duration_s=c.get("ssvep_serial_cue_ssvep_duration_s", 2.0),
            gap_duration_s=c.get("ssvep_serial_gap_duration_s", 2.0),
            mi_duration_s=c.get("ssvep_serial_mi_duration_s", 4.0),
            waveform=c.get("ssvep_serial_waveform", "sine"),
            cue_style=c.get("ssvep_serial_cue_style", "arrow"),
            task_style=c.get("ssvep_serial_task_style", "arrow"),
            display_mode=c.get("ssvep_serial_display_mode", "single_center"),
            stimulus_width=c.get("ssvep_serial_stimulus_width", 0.35),
            stimulus_height=c.get("ssvep_serial_stimulus_height", 0.35),
            border_width=c.get("ssvep_serial_border_width", 4.0),
            dim_opacity=c.get("ssvep_serial_dim_opacity", 0.0),
            arrow_color=c.get("ssvep_serial_arrow_color", "white"),
            arrow_height=c.get("ssvep_serial_arrow_height", 0.20),
        ),
        ssvep_rt=SSVEPRTConfig(
            enabled=c["trial_mode"] == "mi_ssvep_rt",
            mi_enabled=c.get("ssvep_rt_mi_enabled", False),
            mi_checkpoint_path=c.get("ssvep_rt_mi_checkpoint_path", "") or SSVEPRTConfig.mi_checkpoint_path,
            classifier_window_s=c.get("ssvep_rt_classifier_window_s", 1.5),
            classifier_stride_s=c.get("ssvep_rt_classifier_stride_s", 0.25),
            confidence_threshold=c.get("ssvep_rt_confidence_threshold", 0.15),
            left_freq_hz=c.get("ssvep_rt_left_freq_hz", 10.0),
            right_freq_hz=c.get("ssvep_rt_right_freq_hz", 15.0),
            flicker_duration_s=c.get("ssvep_rt_flicker_duration_s", 4.5),
            flicker_mode=c.get("ssvep_rt_flicker_mode", "border"),
            display_mode=c.get("ssvep_rt_display_mode", "single_side"),
            waveform=c.get("ssvep_rt_waveform", "square"),
            flicker_size=tuple(c.get("ssvep_rt_flicker_size", [0.34, 0.34])),
            flicker_y_pos=c.get("ssvep_rt_flicker_y_pos", 0.0),
            left_x_pos=c.get("ssvep_rt_left_x_pos", -0.35),
            right_x_pos=c.get("ssvep_rt_right_x_pos", 0.35),
            bright_color=c.get("ssvep_rt_bright_color", "white"),
            dark_color=c.get("ssvep_rt_dark_color", "black"),
            flicker_border_width=c.get("ssvep_rt_flicker_border_width", 4.0),
            dim_opacity=c.get("ssvep_rt_dim_opacity", 0.0),
            enable_diag=c.get("ssvep_rt_enable_diag", False),
            diag_dir=c.get("ssvep_rt_diag_dir", "diag"),
        ),
        network=NetworkConfig(
            udp_ip=c["udp_ip"],
            udp_port=c["udp_port"],
        ),
        labrecorder=LabRecorderConfig(
            cli_path=_resolve_project_relative_path(
                c.get("labrecorder_cli_path", "LabRecorder/LabRecorderCLI.exe")
            ),
            study_root=c.get("labrecorder_study_root", "D:/CSDIY/EEG/datasets/customs"),
            path_template=c.get("labrecorder_path_template", "%p/%s/%c/%b/run-%n.xdf"),
            auto_record=c.get("labrecorder_auto_record", True),
            stream_queries=tuple(c.get("labrecorder_stream_queries", ('name="obci_eeg1"', 'name="obci_eeg2"'))),
            recording_format=c.get("labrecorder_recording_format", "xdf"),
            csv_marker_mode=c.get("labrecorder_csv_marker_mode", "legacy"),
        ),
        session_cfg=SessionConfig(
            mode=c["mode"],
            class_mode=c["class_mode"],
            trial_mode=c["trial_mode"],
            block_count=c["blocks"] if c.get("blocks") is not None else default_blocks,
            repeats_per_class=c["repeats_per_class"],
            seed=c["seed"],
        ),
        output_dir=_resolve_project_relative_path(c["output_dir"]),
    )


def get_ssvep_target_freq(config: ExperimentConfig, condition: str) -> float:
    if condition == "left":
        return config.ssvep.left_freq_hz
    if condition == "right":
        return config.ssvep.right_freq_hz
    return 0.0


def _build_metadata_payload(config: ExperimentConfig) -> dict[str, Any]:
    """构建范式感知的元数据负载，用于 LSL 流和 JSON 文件。

    仅包含与活跃 trial 模式相关的 section:
    - 所有模式: trial_type, class_mode, participant, session, run, sample_rate,
      channel_names, marker_mapping, mi section
    - mi_ssvep: ssvep section（频率、闪烁模式等）
    - mi_p300: p300 section（SOA、闪烁时长等）
    - ao_mi: ao_mi section（相位时长）
    """
    payload: dict[str, Any] = {
        "trial_type": config.session_cfg.trial_mode,
        "class_mode": config.session_cfg.class_mode,
        "participant": config.participant,
        "session": config.session,
        "run": config.run,
        "sample_rate": config.display.refresh_rate_hz,
        "channel_names": ["C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"],
        "marker_mapping": MARKERS,
        "mi": {
            "imagery_duration_s": config.timings.imagery_s,
            "conditions": list(get_active_conditions(config.session_cfg)),
        },
    }

    if config.session_cfg.trial_mode == "mi_ssvep" or config.ssvep.enabled:
        payload["ssvep"] = {
            "left_freq": config.ssvep.left_freq_hz,
            "right_freq": config.ssvep.right_freq_hz,
            "flicker_mode": config.ssvep.flicker_mode,
            "display_mode": config.ssvep.display_mode,
            "allow_gaze_shift": config.ssvep.allow_gaze_shift,
        }

    if config.session_cfg.trial_mode == "mi_p300" or config.p300.enabled:
        payload["p300"] = {
            "soa_s": config.p300.soa_s,
            "flash_duration_s": config.p300.flash_duration_s,
            "target_probability": config.p300.target_probability,
        }

    if config.session_cfg.trial_mode == "ao_mi":
        payload["ao_mi"] = {
            "ao_prime_duration_s": config.timings.ao_prime_s,
            "ao_mi_duration_s": config.timings.ao_mi_s,
            "mi_only_duration_s": config.timings.mi_only_s,
        }

    if config.session_cfg.trial_mode == "mi_arrow" or config.arrow.enabled:
        payload["mi_arrow"] = {
            "arrow_style": config.arrow.arrow_style,
            "arrow_color": config.arrow.arrow_color,
            "arrow_height": config.arrow.arrow_height,
        }

    if config.session_cfg.trial_mode == "mi_ssvep_arousal" or config.ssvep_arousal.enabled:
        payload["ssvep_arousal"] = {
            "freq_mode": config.ssvep_arousal.freq_mode,
            "fixed_freq_hz": config.ssvep_arousal.fixed_freq_hz,
            "freq_min_hz": config.ssvep_arousal.freq_min_hz,
            "freq_max_hz": config.ssvep_arousal.freq_max_hz,
            "waveform": config.ssvep_arousal.waveform,
            "cue_style": config.ssvep_arousal.cue_style,
            "task_style": config.ssvep_arousal.task_style,
            "flicker_duration_s": config.ssvep_arousal.flicker_duration_s,
            "stimulus_size": config.ssvep_arousal.stimulus_size,
            "dim_opacity": config.ssvep_arousal.dim_opacity,
            "arrow_color": config.ssvep_arousal.arrow_color,
            "arrow_height": config.ssvep_arousal.arrow_height,
        }

    if config.session_cfg.trial_mode == "mi_ssvep_serial" or config.ssvep_serial.enabled:
        payload["ssvep_serial"] = {
            "cue_ssvep_freq_left_hz": config.ssvep_serial.cue_ssvep_freq_left_hz,
            "cue_ssvep_freq_right_hz": config.ssvep_serial.cue_ssvep_freq_right_hz,
            "cue_ssvep_mode": config.ssvep_serial.cue_ssvep_mode,
            "same_freq_hz": config.ssvep_serial.same_freq_hz,
            "cue_ssvep_duration_s": config.ssvep_serial.cue_ssvep_duration_s,
            "gap_duration_s": config.ssvep_serial.gap_duration_s,
            "mi_duration_s": config.ssvep_serial.mi_duration_s,
            "waveform": config.ssvep_serial.waveform,
            "cue_style": config.ssvep_serial.cue_style,
            "task_style": config.ssvep_serial.task_style,
            "display_mode": config.ssvep_serial.display_mode,
            "stimulus_width": config.ssvep_serial.stimulus_width,
            "stimulus_height": config.ssvep_serial.stimulus_height,
            "border_width": config.ssvep_serial.border_width,
            "dim_opacity": config.ssvep_serial.dim_opacity,
            "arrow_color": config.ssvep_serial.arrow_color,
            "arrow_height": config.ssvep_serial.arrow_height,
        }

    return payload


def publish_experiment_metadata(config: ExperimentConfig) -> None:
    """创建一个 LSL StreamOutlet 携带实验元数据，以便 LabRecorder 捕获它。

    推送一个包含所有实验参数 JSON 负载的样本。
    流类型为 experiment_metadata，下游工具可以识别它。
    如果未安装 pylsl，则打印警告并静默返回。
    """
    if pylsl is None:
        print("WARNING: pylsl 未安装，跳过 LSL 元数据流推送（仅写 session_config.json）")
        return

    payload = _build_metadata_payload(config)

    info = pylsl.StreamInfo(
        name="mi_experiment_metadata",
        type="experiment_metadata",
        channel_count=1,
        nominal_srate=0,  # 不规则速率 — 仅一个样本
        channel_format=pylsl.cf_string,
        source_id="mi_experiment_metadata",
    )
    # 以编程方式构建 XML desc — 每个顶级键成为子元素。
    # 嵌套的 dict/list 在元素内序列化为 JSON 字符串。
    # (pylsl 1.18.1 XMLElement 缺少 from_string(); 使用 append_child/append_child_value)
    meta = info.desc().append_child("experiment_metadata")
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            meta.append_child_value(key, json.dumps(value))
        else:
            meta.append_child_value(key, str(value))

    outlet = pylsl.StreamOutlet(info)
    # 推送一个样本以便 LabRecorder 记录此流
    outlet.push_sample([json.dumps(payload)])

    meta_info = f"trial_type={config.session_cfg.trial_mode}"
    if config.ssvep.enabled:
        meta_info += f", ssvep_left={config.ssvep.left_freq_hz}Hz, ssvep_right={config.ssvep.right_freq_hz}Hz"
    if config.p300.enabled:
        meta_info += f", p300_soa={config.p300.soa_s * 1000:.0f}ms"
    print(f"LSL 元数据流已推送: {meta_info}")


def write_session_config_json(config: ExperimentConfig, run_dir: Path) -> Path:
    """将实验元数据写入 session_config.json 作为 XDF 元数据的备份。

    JSON 文件放置在与 XDF 输出相同的目录中，以便
    下游工具（例如 verify_ssvep_signal.py）在
    XDF 文件缺少 experiment_metadata 流时可以回退到它。
    """
    payload = _build_metadata_payload(config)
    path = run_dir / "session_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"session_config.json 已写入: {path}")
    return path


# Trial 随机化：每 block 内 shuffle 所有 trial_specs
def build_blocks(config: ExperimentConfig) -> list[list[Trial]]:
    session_cfg = config.session_cfg
    rng = random.Random(session_cfg.seed)
    blocks: list[list[Trial]] = []
    global_trial_index = 1
    active_conditions = get_active_conditions(session_cfg)
    active_trial_types = get_active_trial_types(session_cfg)

    for block_index in range(1, session_cfg.block_count + 1):
        trial_specs: list[tuple[str, str]] = []
        for trial_type in active_trial_types:
            for condition in active_conditions:
                trial_specs.extend([(trial_type, condition)] * session_cfg.repeats_per_class)
        rng.shuffle(trial_specs)

        block_trials: list[Trial] = []
        for trial_index_in_block, (trial_type, condition) in enumerate(trial_specs, start=1):
            ssvep_target_side = condition if trial_type in ("mi_ssvep", "pure_ssvep", "mi_ssvep_serial", "mi_ssvep_rt") else ""
            p300_target_side = condition if trial_type == "mi_p300" else ""
            # 计算 ssvep_arousal_freq_hz
            ssvep_arousal_freq_hz = 0.0
            if trial_type == "mi_ssvep_arousal":
                if config.ssvep_arousal.freq_mode == "fixed":
                    ssvep_arousal_freq_hz = config.ssvep_arousal.fixed_freq_hz
                else:
                    # 在最小和最大之间随机
                    ssvep_arousal_freq_hz = rng.uniform(config.ssvep_arousal.freq_min_hz, config.ssvep_arousal.freq_max_hz)
            block_trials.append(
                Trial(
                    block_index=block_index,
                    trial_index_in_block=trial_index_in_block,
                    global_trial_index=global_trial_index,
                    condition=condition,
                    trial_type=trial_type,
                    phase_sequence=get_phase_sequence_for_trial_type(trial_type),
                    ssvep_target_side=ssvep_target_side,
                    ssvep_target_freq_hz=(
                        get_ssvep_target_freq(config, condition) if trial_type in ("mi_ssvep", "pure_ssvep")
                        else (config.ssvep_rt.left_freq_hz if condition == "left" else config.ssvep_rt.right_freq_hz) if trial_type == "mi_ssvep_rt"
                        else 0.0
                    ),
                    p300_target_side=p300_target_side,
                    ssvep_arousal_freq_hz=ssvep_arousal_freq_hz,
                )
            )
            global_trial_index += 1
        blocks.append(block_trials)

    return blocks


def serialize_phase_for_plan(phase: PhaseSpec) -> dict[str, object]:
    return {
        "phase_name": phase.phase_name,
        "duration_s": phase.duration_s,
        "screen_kind": phase.screen_kind,
        "title": phase.title,
        "body": phase.body,
        "footer": phase.footer,
        "layout": phase.layout,
        "image_path": str(phase.image_path) if phase.image_path else "",
        "image_scale": phase.image_scale,
        "marker_name": phase.marker_name,
        "marker_value": phase.marker_value if phase.marker_value is not None else "",
        "note": phase.note,
        "center_mode": phase.center_mode,
        "ssvep_target_side": phase.ssvep_target_side,
        "ssvep_target_freq_hz": phase.ssvep_target_freq_hz,
        "ssvep_left_freq_hz": phase.ssvep_left_freq_hz,
        "ssvep_right_freq_hz": phase.ssvep_right_freq_hz,
        "p300_target_side": phase.p300_target_side,
        "p300_soa_s": phase.p300_soa_s,
        "p300_flash_duration_s": phase.p300_flash_duration_s,
        "ssvep_arousal_freq_hz": phase.ssvep_arousal_freq_hz,
        "video_path": str(phase.video_path) if phase.video_path else "",
        "video_start_s": phase.video_start_s,
        "video_flip_horizontal": phase.video_flip_horizontal,
    }


def build_condition_plan(condition: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "cue_marker": CONDITION_TO_CUE_MARKER[condition],
        "pure_mi_marker": CONDITION_TO_MI_MARKER[condition],
        "cue_text": CONDITION_TO_CUE_TEXT[condition],
        "pure_mi_text": CONDITION_TO_TASK_TEXT[condition],
    }
    if condition in {"left", "right"}:
        payload.update(
            {
                "ssvep_marker": MARKERS[f"ssvep_{condition}"],
                "ssvep_gaze_marker": MARKERS[f"ssvep_gaze_{condition}"],
                "ssvep_gaze_text": f"看向{condition}侧闪烁并保持运动想象",
                "p300_target_marker": MARKERS["p300_target_flash"],
                "p300_nontarget_marker": MARKERS["p300_nontarget_flash"],
                "p300_task_text": CONDITION_TO_P300_TASK_TEXT.get(condition, ""),
            }
        )
    if condition in CONDITION_TO_AO_PRIME_MARKER:
        payload.update(
            {
                "ao_prime_marker": CONDITION_TO_AO_PRIME_MARKER[condition],
                "ao_mi_marker": CONDITION_TO_AO_MI_MARKER[condition],
                "mi_only_marker": CONDITION_TO_MI_ONLY_MARKER[condition],
                "ao_prime_text": CONDITION_TO_AO_PRIME_TEXT[condition],
                "ao_mi_text": CONDITION_TO_AO_MI_TEXT[condition],
                "mi_only_text": CONDITION_TO_MI_ONLY_TEXT[condition],
            }
        )
    return payload


def write_session_plan(config: ExperimentConfig, blocks: Sequence[Sequence[Trial]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_conditions = get_active_conditions(config.session_cfg)
    active_trial_types = get_active_trial_types(config.session_cfg)
    payload = {
        "participant": config.participant,
        "session": config.session,
        "run": config.run,
        "timings": asdict(config.timings),
        "display": asdict(config.display),
        "stimuli": {
            "cue_image_path": str(config.stimuli.cue_image_path) if config.stimuli.cue_image_path else "",
            "pure_mi_cue_left_image_path": str(config.stimuli.pure_mi_cue_left_image_path) if config.stimuli.pure_mi_cue_left_image_path else "",
            "pure_mi_cue_right_image_path": str(config.stimuli.pure_mi_cue_right_image_path) if config.stimuli.pure_mi_cue_right_image_path else "",
            "pure_mi_left_image_path": str(config.stimuli.pure_mi_left_image_path) if config.stimuli.pure_mi_left_image_path else "",
            "pure_mi_right_image_path": str(config.stimuli.pure_mi_right_image_path) if config.stimuli.pure_mi_right_image_path else "",
            "pure_mi_rest_image_path": str(config.stimuli.pure_mi_rest_image_path) if config.stimuli.pure_mi_rest_image_path else "",
            "ao_left_image_path": str(config.stimuli.ao_left_image_path) if config.stimuli.ao_left_image_path else "",
            "ao_right_image_path": str(config.stimuli.ao_right_image_path) if config.stimuli.ao_right_image_path else "",
            "image_height": config.stimuli.image_height,
            "task_image_scale": config.stimuli.task_image_scale,
        },
        "ssvep": asdict(config.ssvep),
        "p300": asdict(config.p300),
        "arrow": asdict(config.arrow),
        "ssvep_arousal": asdict(config.ssvep_arousal),
        "ssvep_serial": asdict(config.ssvep_serial),
        "network": asdict(config.network),
        "session_cfg": asdict(config.session_cfg),
        "active_conditions": list(active_conditions),
        "active_trial_types": list(active_trial_types),
        "trial_type_labels": {trial_type: get_trial_type_label(trial_type) for trial_type in active_trial_types},
        "markers": MARKERS,
        "conditions": {
            condition: build_condition_plan(condition)
            for condition in active_conditions
        },
        "blocks": [
            [
                {
                    "block_index": trial.block_index,
                    "trial_index_in_block": trial.trial_index_in_block,
                    "global_trial_index": trial.global_trial_index,
                    "condition": trial.condition,
                    "trial_type": trial.trial_type,
                    "trial_type_label": get_trial_type_label(trial.trial_type),
                     "ssvep_target_side": trial.ssvep_target_side,
                     "ssvep_target_freq_hz": trial.ssvep_target_freq_hz,
                     "p300_target_side": trial.p300_target_side,
                     "ssvep_arousal_freq_hz": trial.ssvep_arousal_freq_hz,
                    "phase_sequence": list(trial.phase_sequence),
                    "phases": [
                        serialize_phase_for_plan(phase)
                        for phase in build_trial_phases(trial, config)
                    ],
                }
                for trial in block
            ]
            for block in blocks
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# waitBlanking=True 垂直同步, useFBO=True 离屏渲染防闪烁
def create_psychopy_window(display: DisplayConfig) -> visual.Window:
    try:
        win = visual.Window(
            size=(display.window_width, display.window_height),
            fullscr=display.fullscreen,
            color=display.background_color,
            units="height",
            allowGUI=not display.fullscreen,
            screen=display.display_index,
            waitBlanking=True,
            useFBO=True,
        )

        # 窗口模式下，PsychoPy 可能忽略 screen 参数并将窗口放在主监视器上。
        # 强制设置窗口位置。
        if not display.fullscreen:
            try:
                from pyglet.canvas import get_display
                pyglet_display = get_display()
                pyglet_screens = pyglet_display.get_screens()
                if 0 <= display.display_index < len(pyglet_screens):
                    target_screen = pyglet_screens[display.display_index]
                    # 将窗口居中于目标屏幕
                    win_x = target_screen.x + (target_screen.width - display.window_width) // 2
                    win_y = target_screen.y + (target_screen.height - display.window_height) // 2
                    
                    # 移动窗口
                    pyglet_win = win.backend.window
                    pyglet_win.set_location(win_x, win_y)
            except Exception:
                pass  # 非关键的，窗口仍然可以工作

        return win
    except Exception as exc:
        print(f"[ERROR] create_psychopy_window failed: {exc!r}")
        import traceback
        traceback.print_exc()
        raise


def create_desktop_window(display: DisplayConfig) -> visual.Window:
    try:
        win = visual.Window(
            size=(display.window_width, display.window_height),
            fullscr=display.fullscreen,
            color=display.background_color,
            units="height",
            allowGUI=True,
            screen=display.display_index,
            winType="glfw",
            waitBlanking=True,
        )
        return win
    except Exception as exc:
        print(f"[ERROR] create_desktop_window failed: {exc!r}")
        import traceback
        traceback.print_exc()
        raise


def create_window(display: DisplayConfig) -> visual.Window:
    if display.window_style == "desktop":
        try:
            return create_desktop_window(display)
        except Exception as exc:
            print(f"Desktop window mode failed, falling back to default PsychoPy window: {exc}")
    return create_psychopy_window(display)


# 10 帧预热 + 120 帧计时测量实际刷新率
def calibrate_refresh_rate(win: visual.Window, n_frames: int = 120) -> float:
    """通过执行计时 flip 循环来测量实际显示器刷新率。

    返回测量到的刷新率（Hz）。窗口应已创建并可见。
    在计时测量前会进行 10 次 flip 的短预热，以让 GPU 管线稳定。
    """
    # 预热翻转以稳定 GPU 管线
    for _ in range(10):
        win.flip()
    # 计时测量
    timer = core.Clock()
    for _ in range(n_frames):
        win.flip()
    elapsed = timer.getTime()
    return n_frames / max(elapsed, 1e-6)


def recreate_window(
    ui: SessionUI,
    *,
    fullscreen: bool | None = None,
    size: tuple[int, int] | None = None,
) -> None:
    current_fullscreen = bool(getattr(ui.win, "_isFullScr", ui.display.fullscreen))
    target_size = size or (ui.display.window_width, ui.display.window_height)
    target_fullscreen = current_fullscreen if fullscreen is None else fullscreen

    # 在窗口重建前释放 AO 视频 — 它们的 OpenGL 纹理
    # 绑定到旧窗口的上下文，在新窗口上会导致崩溃。
    saved_video_path = ui._ao_video_path
    saved_video_start_s = ui._ao_video_start_s
    saved_image_scale = ui.stimuli.task_image_scale
    ui.release_ao_videos()

    new_display = replace(
        ui.display,
        fullscreen=target_fullscreen,
        window_width=target_size[0],
        window_height=target_size[1],
    )
    new_win = create_window(new_display)
    new_win.mouseVisible = False
    setattr(new_win, "windowedSize", target_size)

    old_win = ui.win
    ui.win = new_win
    ui.display = new_display
    # 清除所有 stim 缓存：缓存的对象绑定到旧窗口的
    # OpenGL 上下文（即将被销毁）。在新窗口上绘制它们会导致崩溃。
    ui._text_stim_cache.clear()
    ui._image_stim_cache.clear()
    ui._rect_stim_cache.clear()
    ui._image_draw_size_cache.clear()
    old_win.close()

    # 为新 OpenGL 上下文重新预加载 AO 视频。
    if saved_video_path is not None:
        ui.preload_ao_videos(saved_video_path, saved_video_start_s, saved_image_scale)


def set_windowed_size(ui: SessionUI, size: tuple[int, int]) -> None:
    recreate_window(ui, fullscreen=False, size=size)


def toggle_fullscreen(ui: SessionUI) -> None:
    current = bool(getattr(ui.win, "_isFullScr", ui.display.fullscreen))
    recreate_window(
        ui,
        fullscreen=not current,
        size=(ui.display.window_width, ui.display.window_height),
    )


def handle_runtime_window_hotkeys(ui: SessionUI) -> None:
    keys = event.getKeys(keyList=["f11", "1", "2", "3"])
    if not keys:
        return

    for key in keys:
        if key == "f11":
            toggle_fullscreen(ui)
        elif key in {"1", "2", "3"}:
            preset_index = int(key) - 1
            presets = ui.display.window_size_presets
            if 0 <= preset_index < len(presets):
                set_windowed_size(ui, presets[preset_index])


# 轮询循环 + 重绘：每 10ms 检测一次按键并调用 redraw()
def wait_for_space_or_abort(
    ui: SessionUI | None = None,
    redraw: Callable[[], None] | None = None,
) -> None:
    event.clearEvents()
    while True:
        if ui is not None:
            handle_runtime_window_hotkeys(ui)
        if redraw is not None:
            redraw()
        keys = event.getKeys(keyList=["space", "escape"])
        if not keys:
            core.wait(0.01, hogCPUperiod=0.005)
            continue
        if keys[0] == "escape":
            raise ExperimentAbort()
        return


def wait_for_key_or_abort(
    key_list: list[str],
    ui: SessionUI | None = None,
    redraw: Callable[[], None] | None = None,
) -> str:
    """等待指定按键之一或 ESC 中止。
    
    返回按下的键（小写）。
    """
    # 将 escape 添加到允许的按键列表
    all_keys = list(set(k.lower() for k in key_list) | {"escape"})
    event.clearEvents()
    while True:
        if ui is not None:
            handle_runtime_window_hotkeys(ui)
        if redraw is not None:
            redraw()
        keys = event.getKeys(keyList=all_keys)
        if not keys:
            core.wait(0.01, hogCPUperiod=0.005)
            continue
        pressed = keys[0].lower()
        if pressed == "escape":
            raise ExperimentAbort()
        return pressed


# 轮询循环 + 重绘：每 10ms 检测一次按键并调用 redraw()
def wait_or_abort(
    duration_s: float,
    ui: SessionUI | None = None,
    redraw: Callable[[], None] | None = None,
) -> None:
    timer = core.Clock()
    while timer.getTime() < duration_s:
        if ui is not None:
            handle_runtime_window_hotkeys(ui)
        if redraw is not None:
            redraw()
        if "escape" in event.getKeys(keyList=["escape"]):
            raise ExperimentAbort()
        core.wait(0.01, hogCPUperiod=0.005)


# redraw 闭包模式：绘制后调用 win.flip()
def show_text_screen(
    ui: SessionUI,
    title: str,
    body: str = "",
    footer: str = "",
    image_path: Path | None = None,
    center_drawer: Callable[[], None] | None = None,
    image_scale: float = 1.0,
    layout: str = "info",
) -> Callable[[], None]:
    try:
        def redraw() -> None:
            ui.draw_text_screen(
                title,
                body,
                footer,
                image_path=image_path,
                center_drawer=center_drawer,
                image_scale=image_scale,
                layout=layout,
            )
            ui.win.flip()

        redraw()
        return redraw
    except Exception as exc:
        print(f"[ERROR] show_text_screen failed: {exc!r}")
        import traceback
        traceback.print_exc()
        raise


# 首帧通过 callOnFlip 发送 marker，之后由 redraw 维持画面
def show_timed_text_phase(
    ui: SessionUI,
    title: str,
    body: str,
    footer: str,
    duration_s: float,
    image_path: Path | None = None,
    center_drawer: Callable[[], None] | None = None,
    image_scale: float = 1.0,
    on_flip: Callable[[], None] | None = None,
    layout: str = "stimulus",
) -> None:
    def redraw() -> None:
        ui.draw_text_screen(
            title,
            body,
            footer,
            image_path=image_path,
            center_drawer=center_drawer,
            image_scale=image_scale,
            layout=layout,
        )
        ui.win.flip()

    ui.draw_text_screen(
        title,
        body,
        footer,
        image_path=image_path,
        center_drawer=center_drawer,
        image_scale=image_scale,
        layout=layout,
    )
    if on_flip is not None:
        ui.win.callOnFlip(on_flip)
    ui.win.flip()
    wait_or_abort(duration_s, ui=ui, redraw=redraw)


def show_arrow_cue_phase(
    ui: SessionUI,
    condition: str,
    duration_s: float,
    on_flip: Callable[[], None] | None = None,
) -> None:
    def redraw() -> None:
        ui.draw_arrow_cue(condition)
        ui.win.flip()

    ui.draw_arrow_cue(condition)
    if on_flip is not None:
        ui.win.callOnFlip(on_flip)
    ui.win.flip()
    wait_or_abort(duration_s, ui=ui, redraw=redraw)


def show_arrow_mi_phase(
    ui: SessionUI,
    condition: str,
    duration_s: float,
    on_flip: Callable[[], None] | None = None,
) -> None:
    def redraw() -> None:
        ui.draw_arrow_mi_task(condition)
        ui.win.flip()

    ui.draw_arrow_mi_task(condition)
    if on_flip is not None:
        ui.win.callOnFlip(on_flip)
    ui.win.flip()
    wait_or_abort(duration_s, ui=ui, redraw=redraw)


def show_arousal_cue_phase(
    ui: SessionUI,
    condition: str,
    cue_style: str,
    duration_s: float,
    on_flip: Callable[[], None] | None = None,
) -> None:
    def redraw() -> None:
        ui.draw_arousal_cue(condition, cue_style)
        ui.win.flip()

    ui.draw_arousal_cue(condition, cue_style)
    if on_flip is not None:
        ui.win.callOnFlip(on_flip)
    ui.win.flip()
    wait_or_abort(duration_s, ui=ui, redraw=redraw)


# per-trial 随机频率（18-25Hz）或固定频率中央闪烁
def show_arousal_task_phase(
    ui: SessionUI,
    phase: PhaseSpec,
    condition: str,
    task_style: str,
    waveform: str,
    dim_opacity: float,
    on_flip: Callable[[], None] | None = None,
) -> dict[str, float]:
    frame_index = 0
    timer = core.Clock()
    freq_hz = phase.ssvep_arousal_freq_hz

    _ALL_KEYS = ["f11", "1", "2", "3", "escape"]

    while timer.getTime() < phase.duration_s:
        keys = event.getKeys(keyList=_ALL_KEYS)
        if keys:
            for key in keys:
                if key == "escape":
                    raise ExperimentAbort()
                elif key == "f11":
                    toggle_fullscreen(ui)
                elif key in {"1", "2", "3"}:
                    preset_index = int(key) - 1
                    presets = ui.display.window_size_presets
                    if 0 <= preset_index < len(presets):
                        set_windowed_size(ui, presets[preset_index])

        elapsed_time_s = timer.getTime()
        ui.draw_arousal_task_frame(
            condition,
            task_style,
            elapsed_time_s,
            freq_hz,
            waveform,
            dim_opacity,
        )
        if frame_index == 0 and on_flip is not None:
            ui.win.callOnFlip(on_flip)
        ui.win.flip()
        frame_index += 1

    elapsed_s = max(timer.getTime(), 1e-6)
    return {
        "rendered_frames": float(frame_index),
        "elapsed_s": float(elapsed_s),
        "freq_hz": float(freq_hz),
    }


# 频率编码 vs 同频模式在循环中 fork
def show_serial_ssvep_cue_phase(
    ui: SessionUI,
    phase: PhaseSpec,
    condition: str,
    ssvep_serial: SSVEPSerialConfig,
    on_flip: Callable[[], None] | None = None,
) -> dict[str, float]:
    frame_index = 0
    timer = core.Clock()

    # 根据模式确定频率
    if ssvep_serial.cue_ssvep_mode == "frequency_coded":
        left_freq_hz = ssvep_serial.cue_ssvep_freq_left_hz
        right_freq_hz = ssvep_serial.cue_ssvep_freq_right_hz
    else:  # same_freq
        left_freq_hz = ssvep_serial.same_freq_hz
        right_freq_hz = ssvep_serial.same_freq_hz

    _ALL_KEYS = ["f11", "1", "2", "3", "escape"]

    while timer.getTime() < phase.duration_s:
        keys = event.getKeys(keyList=_ALL_KEYS)
        if keys:
            for key in keys:
                if key == "escape":
                    raise ExperimentAbort()
                elif key == "f11":
                    toggle_fullscreen(ui)
                elif key in {"1", "2", "3"}:
                    preset_index = int(key) - 1
                    presets = ui.display.window_size_presets
                    if 0 <= preset_index < len(presets):
                        set_windowed_size(ui, presets[preset_index])

        elapsed_time_s = timer.getTime()
        ui.draw_serial_ssvep_cue_frame(
            condition,
            ssvep_serial,
            elapsed_time_s,
        )
        if frame_index == 0 and on_flip is not None:
            ui.win.callOnFlip(on_flip)
        ui.win.flip()
        frame_index += 1

    elapsed_s = max(timer.getTime(), 1e-6)
    return {
        "rendered_frames": float(frame_index),
        "elapsed_s": float(elapsed_s),
        "left_hz": float(left_freq_hz),
        "right_hz": float(right_freq_hz),
    }


def show_serial_mi_phase(
    ui: SessionUI,
    condition: str,
    task_style: str,
    duration_s: float,
    on_flip: Callable[[], None] | None = None,
) -> None:
    def redraw() -> None:
        ui.draw_serial_mi_task(condition, task_style)
        ui.win.flip()

    ui.draw_serial_mi_task(condition, task_style)
    if on_flip is not None:
        ui.win.callOnFlip(on_flip)
    ui.win.flip()
    wait_or_abort(duration_s, ui=ui, redraw=redraw)


# 注视点 ○ 渲染
def show_fixation_phase(
    ui: SessionUI,
    duration_s: float,
    on_flip: Callable[[], None] | None = None,
) -> None:
    def redraw() -> None:
        ui.draw_fixation()
        ui.win.flip()

    ui.draw_fixation()
    if on_flip is not None:
        ui.win.callOnFlip(on_flip)
    ui.win.flip()
    wait_or_abort(duration_s, ui=ui, redraw=redraw)


# ⚠ 性能关键：GC 关闭 + frameInterval 关闭 + 时间驱动 flicker + 单次 getKeys/帧
def show_ssvep_phase(
    ui: SessionUI,
    phase: PhaseSpec,
    ssvep: SSVEPConfig,
    on_flip: Callable[[], None] | None = None,
) -> dict[str, float]:
    frame_index = 0
    timer = core.Clock()

    _title = phase.title
    _target_side = phase.ssvep_target_side
    _target_freq_hz = phase.ssvep_target_freq_hz
    _ALL_KEYS = ["f11", "1", "2", "3", "escape"]

    # GC 关闭 + 禁用帧间隔跟踪，在紧密渲染循环期间
    import gc as _gc
    _gc_was_enabled = _gc.isenabled()
    if _gc_was_enabled:
        _gc.disable()
    _was_recording_intervals = ui.win.recordFrameIntervals
    ui.win.recordFrameIntervals = False

    try:
        while timer.getTime() < phase.duration_s:
            keys = event.getKeys(keyList=_ALL_KEYS)
            if keys:
                for key in keys:
                    if key == "escape":
                        raise ExperimentAbort()
                    elif key == "f11":
                        toggle_fullscreen(ui)
                    elif key in {"1", "2", "3"}:
                        preset_index = int(key) - 1
                        presets = ui.display.window_size_presets
                        if 0 <= preset_index < len(presets):
                            set_windowed_size(ui, presets[preset_index])

            elapsed_time_s = timer.getTime()
            ui.draw_ssvep_frame(
                ssvep,
                elapsed_time_s=elapsed_time_s,
                title=_title,
                target_side=_target_side,
                target_freq_hz=_target_freq_hz,
            )
            if frame_index == 0 and on_flip is not None:
                ui.win.callOnFlip(on_flip)
            ui.win.flip()
            frame_index += 1
    finally:
        if _gc_was_enabled:
            _gc.enable()
        ui.win.recordFrameIntervals = _was_recording_intervals

    elapsed_s = max(timer.getTime(), 1e-6)
    return {
        "rendered_frames": float(frame_index),
        "elapsed_s": float(elapsed_s),
        "left_hz": float(ssvep.left_freq_hz),
        "right_hz": float(ssvep.right_freq_hz),
    }


# ⚠ SSVEPConfig-from-SSVEPRTConfig 转换 + LSL 非阻塞反馈叠加
def show_ssvep_rt_phase(
    ui: SessionUI,
    phase: PhaseSpec,
    rt_cfg: SSVEPRTConfig,
    feedback_display: ClassificationFeedbackDisplay,
    on_flip: Callable[[], None] | None = None,
) -> dict[str, float]:
    """SSVEP+MI phase with realtime classification feedback overlay.

    Render-loop design rules for stable framerate:
    - Zero blocking calls (no LSL resolve, no disk I/O, no network)
    - Single event.getKeys per frame (OS event-queue polling is expensive)
    - Pre-connect feedback before entering the loop
    - Pre-compute constant strings/layouts, no per-frame object allocation
    """
    # Build an SSVEPConfig from SSVEPRTConfig for draw_ssvep_frame reuse
    ssvep_for_drawing = SSVEPConfig(
        enabled=True,
        left_freq_hz=rt_cfg.left_freq_hz,
        right_freq_hz=rt_cfg.right_freq_hz,
        flicker_duration_s=rt_cfg.flicker_duration_s,
        allow_gaze_shift=True,
        flicker_size=rt_cfg.flicker_size,
        flicker_y_pos=rt_cfg.flicker_y_pos,
        left_x_pos=rt_cfg.left_x_pos,
        right_x_pos=rt_cfg.right_x_pos,
        flicker_mode=rt_cfg.flicker_mode,
        display_mode=rt_cfg.display_mode,
        waveform=rt_cfg.waveform,
        bright_color=rt_cfg.bright_color,
        dark_color=rt_cfg.dark_color,
        flicker_border_width=rt_cfg.flicker_border_width,
        dim_opacity=rt_cfg.dim_opacity,
    )

    # ── 循环前设置（不要将阻塞调用放入渲染循环中）──
    feedback_display.ensure_connected(timeout=0.1)

    # 预计算本阶段常量字符串
    _title = phase.title
    _target_side = phase.ssvep_target_side
    _target_freq_hz = phase.ssvep_target_freq_hz

    # ── 紧密渲染循环期间关闭 GC ──
    # Python GC 可能暂停 10-50ms，在 16.67ms 帧预算中致命。
    import gc as _gc
    _gc_was_enabled = _gc.isenabled()
    if _gc_was_enabled:
        _gc.disable()

    # ── 为更紧密的循环禁用每帧间隔记录 ──
    _was_recording_intervals = ui.win.recordFrameIntervals
    ui.win.recordFrameIntervals = False

    frame_index = 0
    timer = core.Clock()

    # 单次组合按键列表：热键 + escape 在一次 OS 轮询中完成
    _ALL_KEYS = ["f11", "1", "2", "3", "escape"]

    try:
        while timer.getTime() < phase.duration_s:
            # ── 每帧一次事件轮询（原来是两次）──
            keys = event.getKeys(keyList=_ALL_KEYS)
            if keys:
                for key in keys:
                    if key == "escape":
                        raise ExperimentAbort()
                    elif key == "f11":
                        toggle_fullscreen(ui)
                    elif key in {"1", "2", "3"}:
                        preset_index = int(key) - 1
                        presets = ui.display.window_size_presets
                        if 0 <= preset_index < len(presets):
                            set_windowed_size(ui, presets[preset_index])

            elapsed_time_s = timer.getTime()

            # 绘制 SSVEP 帧（复用现有渲染逻辑）
            ui.draw_ssvep_frame(
                ssvep_for_drawing,
                elapsed_time_s=elapsed_time_s,
                title=_title,
                target_side=_target_side,
                target_freq_hz=_target_freq_hz,
            )

            # 绘制分类反馈文本（仅非阻塞拉取）
            feedback_text = feedback_display.get_feedback_text()
            if feedback_text:
                ui._draw_cached_text(
                    text=feedback_text,
                    pos=(0.0, -0.42),
                    height=0.045,
                    color="yellow",
                )

            if frame_index == 0 and on_flip is not None:
                ui.win.callOnFlip(on_flip)
            ui.win.flip()
            frame_index += 1

    finally:
        if _gc_was_enabled:
            _gc.enable()
        ui.win.recordFrameIntervals = _was_recording_intervals

    elapsed_s = max(timer.getTime(), 1e-6)
    return {
        "rendered_frames": float(frame_index),
        "elapsed_s": float(elapsed_s),
        "left_hz": float(rt_cfg.left_freq_hz),
        "right_hz": float(rt_cfg.right_freq_hz),
    }


# 双侧 cue 画面 + 目标环
def show_dual_cue_phase(
    ui: SessionUI,
    phase: PhaseSpec,
    *,
    left_x_pos: float,
    right_x_pos: float,
    y_pos: float,
    image_height: float,
    target_border_color: str,
    nontarget_border_color: str,
    target_side: str,
    display_mode: str = "both_sides",
    on_flip: Callable[[], None] | None = None,
) -> None:
    def redraw() -> None:
        ui.draw_dual_cue_screen(
            title=phase.title,
            target_side=target_side,
            left_x_pos=left_x_pos,
            right_x_pos=right_x_pos,
            y_pos=y_pos,
            image_height=image_height,
            target_border_color=target_border_color,
            nontarget_border_color=nontarget_border_color,
            display_mode=display_mode,
        )
        ui.win.flip()

    ui.draw_dual_cue_screen(
        title=phase.title,
        target_side=target_side,
        left_x_pos=left_x_pos,
        right_x_pos=right_x_pos,
        y_pos=y_pos,
        image_height=image_height,
        target_border_color=target_border_color,
        nontarget_border_color=nontarget_border_color,
        display_mode=display_mode,
    )
    if on_flip is not None:
        ui.win.callOnFlip(on_flip)
    ui.win.flip()
    wait_or_abort(phase.duration_s, ui=ui, redraw=redraw)



@dataclass(frozen=True)
class _P300FlashEvent:
    """预生成序列中的单个 P300 闪烁事件。"""
    flash_index: int
    onset_s: float
    offset_s: float
    side: str
    is_target: bool
    marker_name: str
    marker_value: int


# 等间距目标分布 + 随机起始相位保证 oddball 意外性
def generate_p300_flash_sequence(
    *,
    duration_s: float,
    soa_s: float,
    flash_duration_s: float,
    target_side: str,
    target_probability: float,
    seed: int,
) -> tuple[_P300FlashEvent, ...]:
    """预生成 P300 闪烁事件序列。

    使用等间距放置和随机起始相位：目标在序列中均匀分布，
    具有保证的最小间隔，同时起始位置随机化以保持不可预测性。

    目标侧闪烁是受关注的 oddball；另一侧是非目标标准。
    """
    rng = random.Random(seed)
    n_flashes = max(1, int(duration_s / soa_s))
    n_target = round(n_flashes * target_probability)
    n_target = max(0, min(n_target, n_flashes))  # 限制到有效范围
    nontarget_side = "right" if target_side == "left" else "left"

    # 等间距放置，随机起始相位：
    # - interval = n_flashes / n_target（精确间隔）
    # - start_offset = random(0, interval-1)
    # - targets at: [start_offset + i*interval for i in range(n_target)]
    # 这保证了相邻目标之间的最小间隔 = interval - 1。
    target_indices: set[int] = set()
    if n_target > 0 and n_flashes > 0:
        interval = n_flashes / n_target
        # 随机起始相位：从 0 到第一个间隔结束前任意位置
        max_start = min(int(interval), n_flashes - n_target + 1)
        start_offset = rng.randint(0, max(0, max_start - 1))
        for i in range(n_target):
            pos = int(start_offset + i * interval)
            pos = min(pos, n_flashes - 1)  # 限制到有效范围
            target_indices.add(pos)

    sequence = []
    for i in range(n_flashes):
        is_target = i in target_indices
        side = target_side if is_target else nontarget_side
        sequence.append(_P300FlashEvent(
            flash_index=i,
            onset_s=i * soa_s,
            offset_s=i * soa_s + flash_duration_s,
            side=side,
            is_target=is_target,
            marker_name="p300_target_flash" if is_target else "p300_nontarget_flash",
            marker_value=MARKERS["p300_target_flash"] if is_target else MARKERS["p300_nontarget_flash"],
        ))
    return tuple(sequence)


# 预生成闪烁序列扫描 + marker 在 onset 时刻发送
def show_p300_phase(
    ui: SessionUI,
    phase: PhaseSpec,
    p300: P300Config,
    sender: UdpMarkerSender,
    trial: Trial,
    logger: EventLogger,
    on_flip: Callable[[], None] | None = None,
) -> dict[str, float]:
    flash_sequence = generate_p300_flash_sequence(
        duration_s=phase.duration_s,
        soa_s=p300.soa_s,
        flash_duration_s=p300.flash_duration_s,
        target_side=phase.p300_target_side,
        target_probability=p300.target_probability,
        seed=p300.flash_sequence_seed + trial.global_trial_index,
    )

    frame_index = 0
    timer = core.Clock()
    next_flash_idx = 0
    markers_sent = 0

    _title = phase.title
    _target_side = phase.p300_target_side
    _ALL_KEYS = ["f11", "1", "2", "3", "escape"]

    while timer.getTime() < phase.duration_s:
        keys = event.getKeys(keyList=_ALL_KEYS)
        if keys:
            for key in keys:
                if key == "escape":
                    raise ExperimentAbort()
                elif key == "f11":
                    toggle_fullscreen(ui)
                elif key in {"1", "2", "3"}:
                    preset_index = int(key) - 1
                    presets = ui.display.window_size_presets
                    if 0 <= preset_index < len(presets):
                        set_windowed_size(ui, presets[preset_index])

        current_time = timer.getTime()

        # 确定当前哪侧正在闪烁
        flashing_side: str | None = None
        for flash in flash_sequence[next_flash_idx:min(next_flash_idx + 2, len(flash_sequence))]:
            if flash.onset_s <= current_time < flash.offset_s:
                flashing_side = flash.side
                break

        # 在闪烁 onset 时刻发送 markers
        while (next_flash_idx < len(flash_sequence)
               and flash_sequence[next_flash_idx].onset_s <= current_time):
            flash = flash_sequence[next_flash_idx]
            if next_flash_idx == 0 and on_flip is not None:
                ui.win.callOnFlip(on_flip)
            sender.send(
                flash.marker_value,
                flash.marker_name,
                block_index=trial.block_index,
                trial_index_in_block=trial.trial_index_in_block,
                global_trial_index=trial.global_trial_index,
                condition=trial.condition,
                trial_type=trial.trial_type,
                phase_name=phase.phase_name,
                note=f"p300_flash; side={flash.side}; is_target={flash.is_target}",
            )
            markers_sent += 1
            next_flash_idx += 1

        ui.draw_p300_frame(
            p300,
            flashing_side=flashing_side,
            title=_title,
            target_side=_target_side,
        )
        ui.win.flip()
        frame_index += 1

    elapsed_s = max(timer.getTime(), 1e-6)
    return {
        "rendered_frames": float(frame_index),
        "elapsed_s": float(elapsed_s),
        "markers_sent": float(markers_sent),
        "total_flashes_planned": float(len(flash_sequence)),
    }



# 优先复用预加载视频（seek + play），无预加载时才创建新 MovieStim
def show_video_phase(
    ui: SessionUI,
    phase: PhaseSpec,
    on_flip: Callable[[], None] | None = None,
) -> None:
    """播放 MP4 视频剪辑，持续相位时长。

    视频寻址到 phase.video_start_s 并播放 phase.duration_s 秒。
    当 phase.video_flip_horizontal 为 True 时视频镜像（用于右手条件，
    因为源视频只包含左手）。

    如果 ui._ao_video_pool 中有预加载的 AO 视频，
    则复用（seek + play）而不是创建新的 MovieStim。
    这消除了每次 trial 创建 MovieStim 造成的数秒阻塞延迟。
    """
    video_path = phase.video_path
    if not video_path or not video_path.exists():
        show_timed_text_phase(
            ui,
            title=phase.title,
            body=phase.body,
            footer=phase.footer,
            duration_s=phase.duration_s,
            image_path=phase.image_path,
            center_drawer=None,
            image_scale=phase.image_scale,
            on_flip=on_flip,
            layout=phase.layout,
        )
        return

    # 尝试使用会话池中的预加载视频。
    movie = ui.get_ao_video(phase.video_flip_horizontal)
    is_preloaded = movie is not None

    if not is_preloaded:
        # 没有预加载的视频；回退到创建新的。
        try:
            movie = visual.MovieStim(
                ui.win,
                str(video_path),
                flipHoriz=phase.video_flip_horizontal,
                pos=(0, -0.02),
                units="height",
                volume=0,
            )
        except Exception:
            # 如果视频无法加载，回退到静态图像。
            show_timed_text_phase(
                ui,
                title=phase.title,
                body=phase.body,
                footer=phase.footer,
                duration_s=phase.duration_s,
                image_path=phase.image_path,
                center_drawer=None,
                image_scale=phase.image_scale,
                on_flip=on_flip,
                layout=phase.layout,
            )
            return

        # 缩放视频以匹配静态图像的相同目标高度。
        target_height = ui.stimuli.image_height * phase.image_scale
        native_size = movie.size
        if native_size[1] > 0:
            aspect_ratio = native_size[0] / native_size[1]
        else:
            aspect_ratio = 16 / 9  # 1280x720 视频的回退
        movie.size = (target_height * aspect_ratio, target_height)

    # 寻址到期望的起始位置。
    movie.pause()
    movie.seek(phase.video_start_s)

    timer = core.Clock()
    movie.play()

    first_flip = True
    while timer.getTime() < phase.duration_s:
        handle_runtime_window_hotkeys(ui)
        if "escape" in event.getKeys(keyList=["escape"]):
            movie.pause()
            raise ExperimentAbort()

        # 绘制文本叠加层（标题、正文、页脚），无静态图像。
        ui.draw_text_screen(
            phase.title,
            phase.body,
            phase.footer,
            image_path=None,
            center_drawer=None,
            image_scale=phase.image_scale,
            layout=phase.layout,
        )

        # 在文本屏幕之上绘制视频帧。
        movie.draw()

        if first_flip and on_flip is not None:
            ui.win.callOnFlip(on_flip)
            first_flip = False

        ui.win.flip()

    # 暂停视频。仅当视频不是预加载的时才卸载 —
    # 预加载的视频跨 trial 复用，在 session 结束时通过 release_ao_videos() 释放。
    movie.pause()
    if not is_preloaded:
        try:
            movie.unload()
        except Exception:
            pass


# 根据 trial_type 构建 PhaseSpec 列表
def build_trial_phases(trial: Trial, config: ExperimentConfig) -> list[PhaseSpec]:
    if trial.trial_type == "mi_ssvep_serial":
        phases = [
            PhaseSpec(
                phase_name="fixation",
                duration_s=config.timings.fixation_s,
                screen_kind="fixation",
                marker_name="fixation_on",
                marker_value=MARKERS["fixation_on"],
                note="fixation_on",
            ),
            PhaseSpec(
                phase_name="serial_ssvep_cue",
                duration_s=config.ssvep_serial.cue_ssvep_duration_s,
                screen_kind="serial_ssvep_cue",
                title=CONDITION_TO_SERIAL_SSVEP_CUE_TEXT[trial.condition],
                marker_name=f"serial_ssvep_cue_{trial.condition}",
                marker_value=CONDITION_TO_SERIAL_SSVEP_CUE_MARKER[trial.condition],
                note="serial_ssvep_cue_on",
            ),
            PhaseSpec(
                phase_name="serial_gap",
                duration_s=config.ssvep_serial.gap_duration_s,
                screen_kind="fixation",
                marker_name="serial_gap",
                marker_value=MARKERS["serial_gap"],
                note="serial_gap_on",
            ),
            PhaseSpec(
                phase_name="serial_mi",
                duration_s=config.ssvep_serial.mi_duration_s,
                screen_kind="serial_mi",
                title=CONDITION_TO_SERIAL_MI_TEXT[trial.condition],
                marker_name=f"serial_mi_{trial.condition}",
                marker_value=CONDITION_TO_SERIAL_MI_MARKER[trial.condition],
                note="serial_mi_on",
            ),
        ]
    elif trial.trial_type == "mi_ssvep_arousal":
        phases = [
            PhaseSpec(
                phase_name="fixation",
                duration_s=config.timings.fixation_s,
                screen_kind="fixation",
                marker_name="fixation_on",
                marker_value=MARKERS["fixation_on"],
                note="fixation_on",
            ),
            PhaseSpec(
                phase_name="arousal_cue",
                duration_s=config.timings.cue_s,
                screen_kind="arousal_cue",
                title=CONDITION_TO_AROUSAL_CUE_TEXT[trial.condition],
                marker_name=f"arousal_cue_{trial.condition}",
                marker_value=CONDITION_TO_AROUSAL_CUE_MARKER[trial.condition],
                note="arousal_cue_on",
            ),
            PhaseSpec(
                phase_name="arousal_task",
                duration_s=config.ssvep_arousal.flicker_duration_s,
                screen_kind="arousal_task",
                title=CONDITION_TO_AROUSAL_TASK_TEXT[trial.condition],
                marker_name=f"arousal_task_{trial.condition}",
                marker_value=CONDITION_TO_AROUSAL_TASK_MARKER[trial.condition],
                note="arousal_task_on",
                ssvep_arousal_freq_hz=trial.ssvep_arousal_freq_hz,
            ),
        ]
    elif trial.trial_type == "mi_ssvep_rt":
        # mi_ssvep_rt 与 mi_ssvep 类似，但使用 rt_ssvep markers（181/182/189）
        # 并包含分类反馈显示
        rt_cfg = config.ssvep_rt
        phases = [
            PhaseSpec(
                phase_name="fixation",
                duration_s=config.timings.fixation_s,
                screen_kind="fixation",
                marker_name="fixation_on",
                marker_value=MARKERS["fixation_on"],
                note="fixation_on",
                ssvep_target_side=trial.ssvep_target_side,
                ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
            ),
            PhaseSpec(
                phase_name="cue",
                duration_s=config.timings.cue_s,
                screen_kind="mi_ssvep_cue",
                title=CONDITION_TO_CUE_TEXT[trial.condition],
                layout="stimulus",
                marker_name=f"cue_{trial.condition}",
                marker_value=CONDITION_TO_CUE_MARKER[trial.condition],
                note=f"cue_on; dual_open_hands; target={trial.ssvep_target_side}",
                center_mode="none",
                ssvep_target_side=trial.ssvep_target_side,
                ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
                ssvep_left_freq_hz=rt_cfg.left_freq_hz,
                ssvep_right_freq_hz=rt_cfg.right_freq_hz,
            ),
            PhaseSpec(
                phase_name="mi_ssvep_rt",
                duration_s=rt_cfg.flicker_duration_s,
                screen_kind="mi_ssvep_rt",
                title=(
                    f"{CONDITION_TO_RT_SSVEP_TEXT[trial.condition]}\n"
                    f"看向{trial.ssvep_target_side}侧闪烁，保持运动想象"
                ),
                marker_name=f"rt_ssvep_{trial.condition}",
                marker_value=CONDITION_TO_RT_SSVEP_MARKER[trial.condition],
                note=f"mi_ssvep_rt_on; dual_fist_flicker; gaze_target={trial.ssvep_target_side}",
                ssvep_target_side=trial.ssvep_target_side,
                ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
                ssvep_left_freq_hz=rt_cfg.left_freq_hz,
                ssvep_right_freq_hz=rt_cfg.right_freq_hz,
            ),
        ]
    elif trial.trial_type == "mi_arrow":
        phases = [
            PhaseSpec(
                phase_name="fixation",
                duration_s=config.timings.fixation_s,
                screen_kind="fixation",
                marker_name="fixation_on",
                marker_value=MARKERS["fixation_on"],
                note="fixation_on",
            ),
            PhaseSpec(
                phase_name="arrow_cue",
                duration_s=config.timings.cue_s,
                screen_kind="arrow_cue",
                title=CONDITION_TO_ARROW_CUE_TEXT[trial.condition],
                marker_name=f"arrow_cue_{trial.condition}",
                marker_value=CONDITION_TO_ARROW_CUE_MARKER[trial.condition],
                note="arrow_cue_on",
            ),
            PhaseSpec(
                phase_name="arrow_mi",
                duration_s=config.timings.imagery_s,
                screen_kind="arrow_mi",
                title=CONDITION_TO_ARROW_MI_TEXT[trial.condition],
                marker_name=f"arrow_mi_{trial.condition}",
                marker_value=CONDITION_TO_ARROW_MI_MARKER[trial.condition],
                note="arrow_mi_on",
            ),
        ]
    else:
        cue_image_path = get_pure_mi_cue_image_path(config.stimuli, trial.condition) if trial.trial_type in ("pure_mi", "ao_mi") else config.stimuli.cue_image_path
        cue_has_image = cue_image_path is not None and cue_image_path.exists()

        if trial.trial_type in ("mi_ssvep", "pure_ssvep"):
            cue_image_path = None
            cue_has_image = True
        elif trial.trial_type == "mi_p300":
            cue_image_path = None
            cue_has_image = True

        phases = [
            PhaseSpec(
                phase_name="fixation",
                duration_s=config.timings.fixation_s,
                screen_kind="fixation",
                marker_name="fixation_on",
                marker_value=MARKERS["fixation_on"],
                note="fixation_on",
                ssvep_target_side=trial.ssvep_target_side,
                ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
            ),
            PhaseSpec(
                phase_name="cue",
                duration_s=config.timings.cue_s,
                screen_kind="mi_ssvep_cue" if trial.trial_type in ("mi_ssvep", "pure_ssvep") else ("mi_p300_cue" if trial.trial_type == "mi_p300" else "text"),
                title=CONDITION_TO_CUE_TEXT[trial.condition],
                layout="stimulus",
                image_path=cue_image_path,
                image_scale=config.stimuli.task_image_scale,
                marker_name=f"cue_{trial.condition}",
                marker_value=CONDITION_TO_CUE_MARKER[trial.condition],
                note="cue_on",
                center_mode="none" if cue_has_image else "cue_cross",
                ssvep_target_side=trial.ssvep_target_side,
                ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
                ssvep_left_freq_hz=config.ssvep.left_freq_hz if trial.trial_type in ("mi_ssvep", "pure_ssvep") else 0.0,
                ssvep_right_freq_hz=config.ssvep.right_freq_hz if trial.trial_type in ("mi_ssvep", "pure_ssvep") else 0.0,
            ),
        ]

        if trial.trial_type == "pure_mi":
            phases.append(
                PhaseSpec(
                    phase_name="pure_mi",
                    duration_s=config.timings.imagery_s,
                    title=CONDITION_TO_TASK_TEXT[trial.condition],
                    layout="stimulus" if trial.condition != "rest" else "title_only",
                    image_path=get_pure_mi_image_path(config.stimuli, trial.condition) if trial.condition != "rest" else None,
                    image_scale=config.stimuli.task_image_scale,
                    marker_name=f"mi_{trial.condition}",
                    marker_value=CONDITION_TO_MI_MARKER[trial.condition],
                    note="pure_mi_on",
                )
            )
        elif trial.trial_type in ("mi_ssvep", "pure_ssvep"):
            phases[1] = PhaseSpec(
                phase_name="cue",
                duration_s=config.timings.cue_s,
                screen_kind="mi_ssvep_cue",
                title=CONDITION_TO_CUE_TEXT[trial.condition],
                layout="stimulus",
                marker_name=f"cue_{trial.condition}",
                marker_value=CONDITION_TO_CUE_MARKER[trial.condition],
                note=f"cue_on; dual_open_hands; target={trial.ssvep_target_side}",
                center_mode="none",
                ssvep_target_side=trial.ssvep_target_side,
                ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
                ssvep_left_freq_hz=config.ssvep.left_freq_hz,
                ssvep_right_freq_hz=config.ssvep.right_freq_hz,
            )
            phases.append(
                PhaseSpec(
                    phase_name="pure_ssvep" if trial.trial_type == "pure_ssvep" else "mi_ssvep",
                    duration_s=config.ssvep.flicker_duration_s,
                    screen_kind="mi_ssvep",
                    title=(
                        f"{CONDITION_TO_TASK_TEXT[trial.condition]}\n"
                        f"看向{trial.ssvep_target_side}侧闪烁"
                    ) if trial.trial_type == "pure_ssvep" else (
                        f"{CONDITION_TO_TASK_TEXT[trial.condition]}\n"
                        f"看向{trial.ssvep_target_side}侧闪烁，保持运动想象"
                    ),
                    marker_name=f"ssvep_{trial.condition}",
                    marker_value=MARKERS[f"ssvep_{trial.condition}"],
                    note=f"ssvep_on; dual_fist_flicker; gaze_target={trial.ssvep_target_side}" if trial.trial_type == "pure_ssvep" else f"mi_ssvep_on; dual_fist_flicker; gaze_target={trial.ssvep_target_side}",
                    ssvep_target_side=trial.ssvep_target_side,
                    ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
                    ssvep_left_freq_hz=config.ssvep.left_freq_hz,
                    ssvep_right_freq_hz=config.ssvep.right_freq_hz,
                )
            )
        elif trial.trial_type == "mi_p300":
            phases[1] = PhaseSpec(
                phase_name="cue",
                duration_s=config.timings.cue_s,
                screen_kind="mi_p300_cue",
                title=CONDITION_TO_CUE_TEXT[trial.condition],
                layout="stimulus",
                marker_name=f"cue_{trial.condition}",
                marker_value=CONDITION_TO_CUE_MARKER[trial.condition],
                note=f"cue_on; dual_open_hands; p300_target={trial.p300_target_side}",
                center_mode="none",
                p300_target_side=trial.p300_target_side,
                p300_soa_s=config.p300.soa_s,
                p300_flash_duration_s=config.p300.flash_duration_s,
            )
            phases.append(
                PhaseSpec(
                    phase_name="mi_p300",
                    duration_s=config.p300.task_duration_s,
                    screen_kind="mi_p300",
                    title=CONDITION_TO_P300_TASK_TEXT[trial.condition],
                    marker_name=f"mi_{trial.condition}",
                    marker_value=CONDITION_TO_MI_MARKER[trial.condition],
                    note=f"mi_p300_on; dual_fist_flash; target={trial.p300_target_side}",
                    p300_target_side=trial.p300_target_side,
                    p300_soa_s=config.p300.soa_s,
                    p300_flash_duration_s=config.p300.flash_duration_s,
                )
            )
        else:
            has_video = config.stimuli.ao_video_path is not None and config.stimuli.ao_video_path.exists()
            phases.extend(
                [
                    PhaseSpec(
                        phase_name="ao_prime",
                        duration_s=config.timings.ao_prime_s,
                        title=CONDITION_TO_AO_PRIME_TEXT[trial.condition],
                        layout="stimulus",
                        screen_kind="ao_video" if has_video else "text",
                        image_path=get_ao_image_path(config.stimuli, trial.condition),
                        video_path=config.stimuli.ao_video_path,
                        video_start_s=config.stimuli.ao_video_start_s,
                        video_flip_horizontal=(trial.condition == "right"),
                        image_scale=config.stimuli.task_image_scale,
                        marker_name=f"ao_prime_{trial.condition}",
                        marker_value=CONDITION_TO_AO_PRIME_MARKER[trial.condition],
                        note="ao_prime_on",
                    ),
                    PhaseSpec(
                        phase_name="ao_mi",
                        duration_s=config.timings.ao_mi_s,
                        title=CONDITION_TO_AO_MI_TEXT[trial.condition],
                        layout="stimulus",
                        screen_kind="ao_video" if has_video else "text",
                        image_path=get_ao_image_path(config.stimuli, trial.condition),
                        video_path=config.stimuli.ao_video_path,
                        video_start_s=config.stimuli.ao_video_start_s,
                        video_flip_horizontal=(trial.condition == "right"),
                        image_scale=config.stimuli.task_image_scale,
                        marker_name=f"ao_mi_{trial.condition}",
                        marker_value=CONDITION_TO_AO_MI_MARKER[trial.condition],
                        note="ao_mi_on",
                    ),
                    PhaseSpec(
                        phase_name="mi_only",
                        duration_s=config.timings.mi_only_s,
                        title=CONDITION_TO_MI_ONLY_TEXT[trial.condition],
                        layout="stimulus",
                        image_path=get_pure_mi_image_path(config.stimuli, trial.condition),
                        image_scale=config.stimuli.task_image_scale,
                        marker_name=f"mi_only_{trial.condition}",
                        marker_value=CONDITION_TO_MI_ONLY_MARKER[trial.condition],
                        note="mi_only_on",
                    ),
                ]
            )

    # ITI 阶段 — mi_ssvep_rt 范式使用 rt_task_off marker
    iti_marker_name = "rt_task_off" if trial.trial_type == "mi_ssvep_rt" else "task_off"
    iti_marker_value = MARKERS[iti_marker_name]
    phases.append(
        PhaseSpec(
            phase_name="iti",
            duration_s=config.timings.iti_s,
            title="RELAX",
            layout="title_only",
            marker_name=iti_marker_name,
            marker_value=iti_marker_value,
            note="inter_trial_interval",
            ssvep_target_side=trial.ssvep_target_side,
            ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
        )
    )

    return phases


def resolve_center_drawer(ui: SessionUI, center_mode: str) -> Callable[[], None] | None:
    if center_mode == "cue_cross":
        return ui.draw_cue_cross
    return None


# on_flip 回调通过 partial 绑定 marker 参数 + screen_kind 分发
def run_phase(
    ui: SessionUI,
    trial: Trial,
    phase: PhaseSpec,
    config: ExperimentConfig,
    logger: EventLogger,
    sender: UdpMarkerSender,
    feedback_display: ClassificationFeedbackDisplay | None = None,
) -> None:
    logger.log_event(
        "phase_start",
        block_index=trial.block_index,
        trial_index_in_block=trial.trial_index_in_block,
        global_trial_index=trial.global_trial_index,
        condition=trial.condition,
        trial_type=trial.trial_type,
        phase_name=phase.phase_name,
        ssvep_target_side=trial.ssvep_target_side,
        ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
        note=phase.note,
    )

    on_flip = None
    if phase.marker_name and phase.marker_value is not None:
        on_flip = partial(
            sender.send,
            phase.marker_value,
            phase.marker_name,
            block_index=trial.block_index,
            trial_index_in_block=trial.trial_index_in_block,
            global_trial_index=trial.global_trial_index,
            condition=trial.condition,
            trial_type=trial.trial_type,
            phase_name=phase.phase_name,
            ssvep_target_side=trial.ssvep_target_side,
            ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
            note=phase.note,
        )

    phase_end_note = phase.note
    if phase.screen_kind == "fixation":
        show_fixation_phase(ui, phase.duration_s, on_flip=on_flip)
    elif phase.screen_kind == "mi_ssvep_cue":
        show_dual_cue_phase(
            ui, phase,
            left_x_pos=config.ssvep.left_x_pos,
            right_x_pos=config.ssvep.right_x_pos,
            y_pos=config.ssvep.flicker_y_pos,
            image_height=config.ssvep.flicker_size[1],
            target_border_color=config.ssvep.target_ring_color,
            nontarget_border_color=config.ssvep.cue_nontarget_border_color,
            target_side=phase.ssvep_target_side,
            display_mode=config.ssvep.display_mode,
            on_flip=on_flip,
        )
    elif phase.screen_kind == "mi_p300_cue":
        show_dual_cue_phase(
            ui, phase,
            left_x_pos=config.p300.left_x_pos,
            right_x_pos=config.p300.right_x_pos,
            y_pos=config.p300.y_pos,
            image_height=config.p300.image_size[1],
            target_border_color=config.p300.cue_target_border_color,
            nontarget_border_color=config.p300.cue_nontarget_border_color,
            target_side=phase.p300_target_side,
            on_flip=on_flip,
        )
    elif phase.screen_kind == "mi_ssvep":
        # SSVEP stats: 记录 rendered_frames / elapsed_s / freq_hz 用于帧率分析
        stats = show_ssvep_phase(ui, phase, config.ssvep, on_flip=on_flip)
        stats_note = (
            f"rendered_frames={int(stats['rendered_frames'])}; "
            f"elapsed_s={stats['elapsed_s']:.4f}; "
            f"left_hz={stats['left_hz']:.4f}; "
            f"right_hz={stats['right_hz']:.4f}"
        )
        logger.log_event(
            "ssvep_render_stats",
            block_index=trial.block_index,
            trial_index_in_block=trial.trial_index_in_block,
            global_trial_index=trial.global_trial_index,
            condition=trial.condition,
            trial_type=trial.trial_type,
            phase_name=phase.phase_name,
            ssvep_target_side=trial.ssvep_target_side,
            ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
            note=stats_note,
        )
        phase_end_note = f"{phase.note}; {stats_note}"
    elif phase.screen_kind == "mi_ssvep_rt":
        # 使用 feedback_display（如果可用），否则创建 dummy
        fb = feedback_display if feedback_display is not None else ClassificationFeedbackDisplay()
        fb.clear()
        stats = show_ssvep_rt_phase(ui, phase, config.ssvep_rt, fb, on_flip=on_flip)
        stats_note = (
            f"rendered_frames={int(stats['rendered_frames'])}; "
            f"elapsed_s={stats['elapsed_s']:.4f}; "
            f"left_hz={stats['left_hz']:.4f}; "
            f"right_hz={stats['right_hz']:.4f}"
        )
        logger.log_event(
            "ssvep_rt_render_stats",
            block_index=trial.block_index,
            trial_index_in_block=trial.trial_index_in_block,
            global_trial_index=trial.global_trial_index,
            condition=trial.condition,
            trial_type=trial.trial_type,
            phase_name=phase.phase_name,
            ssvep_target_side=trial.ssvep_target_side,
            ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
            note=stats_note,
        )
        phase_end_note = f"{phase.note}; {stats_note}"
    elif phase.screen_kind == "mi_p300":
        stats = show_p300_phase(ui, phase, config.p300, sender, trial, logger, on_flip=on_flip)
        stats_note = (
            f"rendered_frames={int(stats['rendered_frames'])}; "
            f"elapsed_s={stats['elapsed_s']:.4f}; "
            f"markers_sent={int(stats['markers_sent'])}; "
            f"total_flashes_planned={int(stats['total_flashes_planned'])}"
        )
        logger.log_event(
            "p300_render_stats",
            block_index=trial.block_index,
            trial_index_in_block=trial.trial_index_in_block,
            global_trial_index=trial.global_trial_index,
            condition=trial.condition,
            trial_type=trial.trial_type,
            phase_name=phase.phase_name,
            note=stats_note,
        )
        phase_end_note = f"{phase.note}; {stats_note}"
    elif phase.screen_kind == "ao_video":
        show_video_phase(ui, phase, on_flip=on_flip)
    elif phase.screen_kind == "arrow_cue":
        show_arrow_cue_phase(ui, trial.condition, phase.duration_s, on_flip=on_flip)
    elif phase.screen_kind == "arrow_mi":
        show_arrow_mi_phase(ui, trial.condition, phase.duration_s, on_flip=on_flip)
    elif phase.screen_kind == "arousal_cue":
        show_arousal_cue_phase(ui, trial.condition, config.ssvep_arousal.cue_style, phase.duration_s, on_flip=on_flip)
    elif phase.screen_kind == "arousal_task":
        stats = show_arousal_task_phase(ui, phase, trial.condition, config.ssvep_arousal.task_style, config.ssvep_arousal.waveform, config.ssvep_arousal.dim_opacity, on_flip=on_flip)
        stats_note = (
            f"rendered_frames={int(stats['rendered_frames'])}; "
            f"elapsed_s={stats['elapsed_s']:.4f}; "
            f"freq_hz={stats['freq_hz']:.4f}"
        )
        logger.log_event(
            "ssvep_arousal_render_stats",
            block_index=trial.block_index,
            trial_index_in_block=trial.trial_index_in_block,
            global_trial_index=trial.global_trial_index,
            condition=trial.condition,
            trial_type=trial.trial_type,
            phase_name=phase.phase_name,
            note=stats_note,
        )
    elif phase.screen_kind == "serial_ssvep_cue":
        stats = show_serial_ssvep_cue_phase(ui, phase, trial.condition, config.ssvep_serial, on_flip=on_flip)
        stats_note = (
            f"rendered_frames={int(stats['rendered_frames'])}; "
            f"elapsed_s={stats['elapsed_s']:.4f}; "
            f"left_hz={stats['left_hz']:.4f}; "
            f"right_hz={stats['right_hz']:.4f}"
        )
        logger.log_event(
            "ssvep_serial_cue_render_stats",
            block_index=trial.block_index,
            trial_index_in_block=trial.trial_index_in_block,
            global_trial_index=trial.global_trial_index,
            condition=trial.condition,
            trial_type=trial.trial_type,
            phase_name=phase.phase_name,
            note=stats_note,
        )
        phase_end_note = f"{phase.note}; {stats_note}"
    elif phase.screen_kind == "serial_mi":
        show_serial_mi_phase(ui, trial.condition, config.ssvep_serial.task_style, phase.duration_s, on_flip=on_flip)
    else:
        show_timed_text_phase(
            ui,
            title=phase.title,
            body=phase.body,
            footer=phase.footer,
            duration_s=phase.duration_s,
            image_path=phase.image_path,
            center_drawer=resolve_center_drawer(ui, phase.center_mode),
            image_scale=phase.image_scale,
            on_flip=on_flip,
            layout=phase.layout,
        )

    logger.log_event(
        "phase_end",
        block_index=trial.block_index,
        trial_index_in_block=trial.trial_index_in_block,
        global_trial_index=trial.global_trial_index,
        condition=trial.condition,
        trial_type=trial.trial_type,
        phase_name=phase.phase_name,
        ssvep_target_side=trial.ssvep_target_side,
        ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
        note=phase_end_note,
    )


# Trial 生命周期：trial_start → phase 循环 → trial_end
def run_trial(
    ui: SessionUI,
    trial: Trial,
    config: ExperimentConfig,
    logger: EventLogger,
    sender: UdpMarkerSender,
    feedback_display: ClassificationFeedbackDisplay | None = None,
) -> None:
    logger.log_event(
        "trial_start",
        block_index=trial.block_index,
        trial_index_in_block=trial.trial_index_in_block,
        global_trial_index=trial.global_trial_index,
        condition=trial.condition,
        trial_type=trial.trial_type,
        ssvep_target_side=trial.ssvep_target_side,
        ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
        note="|".join(trial.phase_sequence),
    )

    for phase in build_trial_phases(trial, config):
        run_phase(ui, trial, phase, config, logger, sender, feedback_display=feedback_display)

    logger.log_event(
        "trial_end",
        block_index=trial.block_index,
        trial_index_in_block=trial.trial_index_in_block,
        global_trial_index=trial.global_trial_index,
        condition=trial.condition,
        trial_type=trial.trial_type,
        ssvep_target_side=trial.ssvep_target_side,
        ssvep_target_freq_hz=trial.ssvep_target_freq_hz,
        note="|".join(trial.phase_sequence),
    )


# ⚠ 总控流程：窗口→校准→录制→block/trial 循环→停止录制→帧诊断
def run_session(config: ExperimentConfig) -> int:
    blocks = build_blocks(config)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_name = f"{config.participant}_{config.session}_run-{config.run:02d}_{config.session_cfg.mode}_{timestamp}"
    run_dir = config.output_dir / base_name
    csv_path = run_dir / f"{base_name}_events.csv"
    json_path = run_dir / f"{base_name}_plan.json"
    write_session_plan(config, blocks, json_path)

    # 通过 LSL 流发布实验元数据 + 写入备份 JSON
    publish_experiment_metadata(config)
    write_session_config_json(config, run_dir)

    # ── 在开始录制前检查 LSL 流 ──
    lsl_check = _check_lsl_streams(config.labrecorder.stream_queries)
    print(f"LSL 流检测: {lsl_check['message']}")
    for result in lsl_check["results"]:
        status = "[OK]" if result["found"] else "[X]"
        print(f"  {status} {result['query']}")
        if result["found"] and result["stream_info"]:
            info = result["stream_info"]
            print(f"      → name={info['name']}, type={info['type']}, "
                  f"ch={info['channel_count']}, rate={info['nominal_srate']}Hz")

    recording_format = config.labrecorder.recording_format
    print(f"录制格式: {recording_format} (marker_mode={config.labrecorder.csv_marker_mode})")

    # ── 启动 LabRecorderCLI 进行自动 XDF 录制 ──
    lr = LabRecorderCLIController(config.labrecorder)
    xdf_path = ""
    if config.labrecorder.auto_record and recording_format in ("xdf", "both"):
        try:
            xdf_path = lr.start_recording(
                participant=config.participant,
                session=config.session,
                run=config.run,
                task=config.session_cfg.trial_mode,
                class_mode=config.session_cfg.class_mode,
            )
        except Exception as exc:
            print(f"WARNING: LabRecorderCLI 启动失败: {exc}")
            print("将不录制 XDF 文件。如需录制，请手动启动 LabRecorder。")

    # ── 启动 LslCsvRecorder 进行 CSV 录制 ──
    csv_recorder: LslCsvRecorder | None = None
    raw_csv_path = ""
    if config.labrecorder.auto_record and recording_format in ("csv", "both"):
        # 在与 XDF 相同的目录结构中构建 CSV 路径
        # 解析路径模板但将 .xdf 替换为 .csv
        from datetime import datetime
        csv_path_template = config.labrecorder.path_template
        csv_path_str = config.labrecorder.study_root.rstrip("/")
        csv_path_str += "/" + csv_path_template.replace("%p", config.participant)
        csv_path_str = csv_path_str.replace("%s", config.session)
        csv_path_str = csv_path_str.replace("%c", config.session_cfg.class_mode)
        csv_path_str = csv_path_str.replace("%b", config.session_cfg.trial_mode)
        csv_path_str = csv_path_str.replace("%n", str(config.run))
        csv_path_str = csv_path_str.replace("%d", datetime.now().strftime("%m%d"))
        csv_path_str = csv_path_str.replace(".xdf", ".csv")
        # 传统命名：mi + 时间戳
        legacy_csv_name = f"mi{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        # 放置在与 XDF 相同的目录中
        raw_csv_path = csv_path_str
        try:
            csv_recorder = LslCsvRecorder(
                csv_path=raw_csv_path,
                marker_mode=config.labrecorder.csv_marker_mode,
            )
            csv_recorder.start_recording()
        except Exception as exc:
            print(f"WARNING: LslCsvRecorder 启动失败: {exc}")
            print("将不录制 CSV 文件。")
            csv_recorder = None

    logger = EventLogger(csv_path, config)
    sender = UdpMarkerSender(config.network, logger)
    win = create_window(config.display)
    setattr(win, "windowedSize", (config.display.window_width, config.display.window_height))
    ui = SessionUI(win, config.stimuli, config.display)
    ui.win.mouseVisible = False

    # ── 启用每帧计时诊断 ──
    win.recordFrameIntervals = True
    win.refreshThreshold = 1.0 / max(config.display.refresh_rate_hz, 1e-6) + 0.004

    # ── 校准实际刷新率 ──
    measured_hz = calibrate_refresh_rate(win)
    config_hz = config.display.refresh_rate_hz
    hz_deviation_pct = abs(measured_hz - config_hz) / max(config_hz, 1e-6) * 100.0
    print(f"Refresh rate: configured={config_hz:.1f} Hz, measured={measured_hz:.1f} Hz, "
          f"deviation={hz_deviation_pct:.1f}%")
    # 实际刷新率偏离 >5% 时警告 VSync 可能未生效
    if hz_deviation_pct > 5.0:
        print(
            f"INFO: measured refresh rate ({measured_hz:.1f} Hz) deviates >5% from "
            f"configured value ({config_hz:.1f} Hz). This indicates VSync is not "
            f"working properly. SSVEP flicker frequencies are still accurate "
            f"(time-driven sinusoidal), but frame drops will reduce visual quality."
        )

    # 为 FPS 诊断用测量值更新 ui.display
    ui.display = replace(ui.display, refresh_rate_hz=measured_hz)

    # 使用时间驱动正弦 flicker，实际 SSVEP 频率
    # 等于配置频率，与帧率无关。
    predicted_left_hz = config.ssvep.left_freq_hz
    predicted_right_hz = config.ssvep.right_freq_hz

    # 如果 session 使用 ao_mi trial，预加载 AO 视频，
    # 以便 show_video_phase() 可以复用它们，而不是在每个
    # trial 上阻塞创建新的 MovieStim。
    if "ao_mi" in get_active_trial_types(config.session_cfg):
        ui.preload_ao_videos(
            config.stimuli.ao_video_path,
            config.stimuli.ao_video_start_s,
            config.stimuli.task_image_scale,
        )

    # ── 启动分类器子进程并连接 mi_ssvep_rt 的反馈显示 ──
    classifier_manager: RTClassifierManager | None = None
    feedback_display: ClassificationFeedbackDisplay | None = None
    if config.session_cfg.trial_mode == "mi_ssvep_rt":
        classifier_manager = RTClassifierManager(config)
        try:
            classifier_manager.start()
            _rt_logger.info("mi_ssvep_rt: classifier subprocess started")
        except Exception as exc:
            _rt_logger.warning("classifier subprocess failed to start: %s", exc)
        feedback_display = ClassificationFeedbackDisplay()
        try:
            connected = feedback_display.try_connect(timeout=5.0)
            if connected:
                _rt_logger.info("mi_ssvep_rt: connected to classification_result LSL stream")
            else:
                _rt_logger.warning("mi_ssvep_rt: could not connect to classification_result LSL stream")
        except Exception as exc:
            _rt_logger.warning("mi_ssvep_rt: feedback display connection error: %s", exc)

    lr_status_parts = []
    if recording_format in ("xdf", "both"):
        lr_status_parts.append(f"XDF [{'OK' if lr.is_recording else 'OFF'}] {xdf_path}" if lr.is_recording else "XDF [OFF] not recording")
    if recording_format in ("csv", "both"):
        lr_status_parts.append(f"CSV [{'OK' if csv_recorder and csv_recorder.is_recording else 'OFF'}] {raw_csv_path}" if csv_recorder and csv_recorder.is_recording else "CSV [OFF] not recording")
    lr_status = " | ".join(lr_status_parts) if lr_status_parts else "Recording [OFF]"

    # 构建刷新率状态行用于显示
    if hz_deviation_pct > 2.0:
        refresh_status = (
            f"Refresh: {measured_hz:.1f} Hz (config: {config_hz:.1f} Hz, "
            f"dev: {hz_deviation_pct:.1f}%)"
        )
    else:
        refresh_status = f"Refresh: {measured_hz:.1f} Hz"

    # 构建 SSVEP 频率状态行
    ssvep_freq_status = (
        f"SSVEP: L {config.ssvep.left_freq_hz:.1f} Hz / R {config.ssvep.right_freq_hz:.1f} Hz"
    )
    if hz_deviation_pct > 2.0:
        ssvep_freq_status += (
            f"  ->  actual: L {predicted_left_hz:.2f} Hz / R {predicted_right_hz:.2f} Hz"
        )

    logger.log_event(
        "session_initialized",
        note=(
            f"plan={json_path.name}; udp={config.network.udp_ip}:{config.network.udp_port}; "
            f"refresh_config={config_hz:.1f}Hz; refresh_measured={measured_hz:.1f}Hz; "
            f"ssvep={config.ssvep.left_freq_hz:.1f}/{config.ssvep.right_freq_hz:.1f}Hz; "
            f"ssvep_predicted={predicted_left_hz:.2f}/{predicted_right_hz:.2f}Hz; "
            f"flicker_mode={config.ssvep.flicker_mode}; "
            f"gaze_shift={config.ssvep.allow_gaze_shift}; "
            f"p300_soa={config.p300.soa_s*1000:.0f}ms; "
            f"p300_flash={config.p300.flash_duration_s*1000:.0f}ms; "
            f"p300_mode={config.p300.flash_mode}; "
            f"{lr_status}"
        ),
    )

    try:
        total_trials = sum(len(block) for block in blocks)
        active_conditions = get_active_conditions(config.session_cfg)
        active_trial_types = get_active_trial_types(config.session_cfg)

        # 构建 LSL 状态行用于显示
        lsl_status_line = lsl_check["message"]
        if not lsl_check["all_found"]:
            missing = [r["query"] for r in lsl_check["results"] if not r["found"]]
            lsl_status_line = f"[!] LSL missing: {', '.join(missing)}"

        # 构建范式特定信息行
        paradigm_info_lines = []
        trial_mode = config.session_cfg.trial_mode
        if trial_mode == "mi_ssvep":
            paradigm_info_lines.append(
                f"SSVEP: L={config.ssvep.left_freq_hz:.1f} Hz, R={config.ssvep.right_freq_hz:.1f} Hz  "
                f"({config.ssvep.flicker_mode})"
            )
        elif trial_mode == "mi_p300":
            paradigm_info_lines.append(
                f"P300: target prob={config.p300.target_probability*100:.0f}%  "
                f"({config.p300.flash_mode})"
            )
        elif trial_mode == "ao_mi":
            paradigm_info_lines.append("AO+MI: video + motor imagery")

        start_redraw = show_text_screen(
            ui,
            title="Motor Imagery Session",
            body=(
                f"Participant: {config.participant}  |  "
                f"Session: {config.session}  |  Run: {config.run}\n"
                f"Trial mode: {trial_mode}  |  Mode: {config.session_cfg.mode}  |  "
                f"Classes: {', '.join(active_conditions)}\n"
                f"Trials: {total_trials}\n"
                + ("\n".join(paradigm_info_lines) + "\n" if paradigm_info_lines else "")
                + f"\n{refresh_status}\n"
                f"{lsl_status_line}\n"
                f"{lr_status}\n\n"
                "SPACE = start  |  ESC = abort"
            ),
            layout="info",
        )
        wait_for_space_or_abort(
            ui=ui,
            redraw=start_redraw,
        )

        for block in blocks:
            block_index = block[0].block_index
            block_redraw = show_text_screen(
                ui,
                title=f"Block {block_index}/{config.session_cfg.block_count}",
                body=(
                    f"{len(block)} trials\n\n"
                    "SPACE to continue"
                ),
                layout="info",
            )
            wait_for_space_or_abort(
                ui=ui,
                redraw=block_redraw,
            )
            sender.send(
                MARKERS["block_start"],
                "block_start",
                block_index=block_index,
                note="block entered",
            )
            logger.log_event("block_start", block_index=block_index)

            for trial in block:
                # 在每个 trial 前检查录制状态
                if config.labrecorder.auto_record and lr.is_recording and recording_format in ("xdf", "both"):
                    is_ok, error_msg = lr.check_recording_status()
                    if not is_ok:
                        # 录制丢失 — 警告用户
                        warn_redraw = show_text_screen(
                            ui,
                            title="Recording Lost",
                            body=(
                                f"LabRecorderCLI has stopped unexpectedly!\n\n"
                                f"{error_msg}\n\n"
                                f"Press SPACE to abort session,\n"
                                f"or press C to continue without recording."
                            ),
                        )
                        choice = wait_for_key_or_abort(
                            ui=ui,
                            redraw=warn_redraw,
                            key_list=["space", "c"],
                        )
                        if choice == "space":
                            raise ExperimentAbort("Recording lost - user chose to abort")
                        # 用户选择继续而不录制
                        logger.log_event(
                            "recording_lost",
                            block_index=trial.block_index,
                            trial_index_in_block=trial.trial_index_in_block,
                            note="user chose to continue without recording",
                        )
                
                run_trial(ui, trial, config, logger, sender, feedback_display=feedback_display)

            sender.send(
                MARKERS["block_end"],
                "block_end",
                block_index=block_index,
                note="block finished",
            )
            logger.log_event("block_end", block_index=block_index)

            # 清除 block 间的图片缓存以防止 GPU 纹理
            # 内存堆积（尤其是动画帧序列）。
            ui._image_stim_cache.clear()
            ui._image_draw_size_cache.clear()

            if block_index < config.session_cfg.block_count:
                break_redraw = show_text_screen(
                    ui,
                    title="Break",
                    body="Let the participant relax for a moment.\nPress SPACE to continue to the next block.",
                )
                wait_for_space_or_abort(
                    ui=ui,
                    redraw=break_redraw,
                )

        sender.send(MARKERS["session_end"], "session_end", note="session complete")
        logger.log_event("session_end", note="session complete")

        # 停止 LabRecorderCLI — XDF 文件已完成
        lr.stop_recording()

        # 停止 CSV 录制器 — CSV 文件已完成
        if csv_recorder is not None:
            csv_recorder.stop_recording()

        # 检查 XDF 文件是否实际有数据
        xdf_ok = xdf_path and Path(xdf_path).exists() and Path(xdf_path).stat().st_size > 0
        if xdf_ok:
            xdf_msg = f"XDF saved to: {xdf_path}"
        elif xdf_path:
            xdf_msg = f"XDF file is empty: {xdf_path}\n(no LSL streams were available — was OpenBCI GUI running?)"
        else:
            xdf_msg = "No XDF was recorded (LabRecorderCLI not started)."

        # 检查 CSV 文件是否实际有数据
        csv_ok = raw_csv_path and Path(raw_csv_path).exists() and Path(raw_csv_path).stat().st_size > 0
        csv_samples = csv_recorder.samples_written if csv_recorder is not None else 0
        if csv_ok:
            csv_msg = f"CSV saved to: {raw_csv_path}\n({csv_samples} samples, marker_mode={config.labrecorder.csv_marker_mode})"
        elif raw_csv_path:
            csv_msg = f"CSV file is empty: {raw_csv_path}\n(no LSL streams were available — was OpenBCI GUI running?)"
        else:
            csv_msg = ""

        # 构建组合录制状态消息
        recording_msgs = []
        if recording_format in ("xdf", "both"):
            recording_msgs.append(xdf_msg)
        if recording_format in ("csv", "both") and csv_msg:
            recording_msgs.append(csv_msg)
        recording_msg = "\n".join(recording_msgs) if recording_msgs else "No data recording was configured."

        complete_redraw = show_text_screen(
            ui,
            title="Session complete",
            body=(
                "The local plan and event logs have been saved.\n"
                + recording_msg
            ),
        )
        wait_for_space_or_abort(
            ui=ui,
            redraw=complete_redraw,
        )
        return 0

    except ExperimentAbort:
        logger.log_event("session_aborted", note="ESC pressed by operator")
        # ESC 时停止 LabRecorderCLI — XDF 文件保存到此点为止
        lr.stop_recording()
        # ESC 时停止 CSV 录制器 — CSV 文件保存到此点为止
        if csv_recorder is not None:
            csv_recorder.stop_recording()
        # 检查 XDF 文件是否实际有数据
        xdf_ok = xdf_path and Path(xdf_path).exists() and Path(xdf_path).stat().st_size > 0
        if xdf_ok:
            xdf_msg = (
                f"XDF saved to: {xdf_path}\n\n"
                "Check the local CSV log and decide whether the XDF should be discarded."
            )
        elif xdf_path:
            xdf_msg = (
                f"XDF file is empty: {xdf_path}\n"
                "(no LSL streams were available — was OpenBCI GUI running?)\n\n"
                "Check the local CSV log and decide whether the data should be discarded."
            )
        else:
            xdf_msg = "Check the local CSV log and decide whether the data should be discarded."

        # 检查原始 CSV 文件是否实际有数据
        csv_ok = raw_csv_path and Path(raw_csv_path).exists() and Path(raw_csv_path).stat().st_size > 0
        csv_msg = ""
        if csv_ok and csv_recorder is not None:
            csv_msg = f"\nCSV saved to: {raw_csv_path} ({csv_recorder.samples_written} samples)"
        elif raw_csv_path:
            csv_msg = f"\nCSV file is empty: {raw_csv_path}"

        aborted_redraw = show_text_screen(
            ui,
            title="Session aborted",
            body="The session was interrupted safely.\n" + xdf_msg + csv_msg,
        )
        event.clearEvents()
        while True:
            handle_runtime_window_hotkeys(ui)
            aborted_redraw()
            keys = event.getKeys(keyList=["space"])
            if keys:
                break
            core.wait(0.01, hogCPUperiod=0.005)
        return 1

    finally:
        # ── 输出帧间隔统计（均值/标准差/丢帧数）──
        try:
            fi = win.frameIntervals
            if fi:
                import numpy as _np
                fi_arr = _np.array(fi)
                print(
                    f"Frame intervals: "
                    f"n={len(fi_arr)}, "
                    f"mean={fi_arr.mean()*1000:.2f}ms, "
                    f"std={fi_arr.std()*1000:.3f}ms, "
                    f"min={fi_arr.min()*1000:.2f}ms, "
                    f"max={fi_arr.max()*1000:.2f}ms, "
                    f"dropped={win.nDroppedFrames}"
                )
                logger.log_event(
                    "frame_diagnostics",
                    note=(
                        f"n={len(fi_arr)}; "
                        f"mean_ms={fi_arr.mean()*1000:.2f}; "
                        f"std_ms={fi_arr.std()*1000:.3f}; "
                        f"min_ms={fi_arr.min()*1000:.2f}; "
                        f"max_ms={fi_arr.max()*1000:.2f}; "
                        f"dropped={win.nDroppedFrames}"
                    ),
                )
        except Exception:
            pass
        # 确保分类器子进程无论退出路径如何都已停止
        try:
            if classifier_manager is not None:
                classifier_manager.stop()
        except Exception:
            pass
        # 确保 LabRecorderCLI 无论退出路径如何都已停止
        try:
            lr.stop_recording()
        except Exception:
            pass
        # 确保 CSV 录制器无论退出路径如何都已停止
        try:
            if csv_recorder is not None:
                csv_recorder.stop_recording()
        except Exception:
            pass
        try:
            sender.close()
        except Exception:
            pass
        try:
            logger.close()
        except Exception:
            pass
        try:
            ui.release_ao_videos()
        except Exception:
            pass
        try:
            ui.win.close()
        except Exception:
            pass
        core.quit()


# ---------------------------------------------------------------------------
# Dialog helpers
# ---------------------------------------------------------------------------

_LAST_SESSION_FILE = Path(__file__).resolve().parent / ".last_session.json"


def _detect_monitors() -> list[dict[str, Any]]:
    """检测连接的显示器及其刷新率。

    使用 Pyglet（PsychoPy 的后端）枚举屏幕，确保
    屏幕索引与 PsychoPy 的 Window(screen=...) 参数匹配。

    返回包含以下键的字典列表：index, label, refresh_rate, is_primary。
    出错时回退到单个 60 Hz 显示器。
    """
    monitors: list[dict[str, Any]] = []
    try:
        # 直接使用 Pyglet — 这确保屏幕索引与 PsychoPy 匹配
        from pyglet.canvas import get_display
        pyglet_display = get_display()
        pyglet_screens = pyglet_display.get_screens()

        # 尝试确定主屏幕（通常位于原点 0,0）
        primary_idx = 0
        for i, scr in enumerate(pyglet_screens):
            if scr.x == 0 and scr.y == 0:
                primary_idx = i
                break

        for idx, scr in enumerate(pyglet_screens):
            # 获取刷新率（如果可用）
            refresh = 60
            try:
                mode = scr.get_mode()
                if mode and hasattr(mode, 'rate') and mode.rate > 0:
                    refresh = int(mode.rate)
            except Exception:
                pass

            is_primary = (idx == primary_idx)
            label = f"显示器 {idx + 1}"
            if is_primary:
                label += " (主屏)"
            label += f" [{scr.width}x{scr.height}]"

            monitors.append({
                "index": idx,
                "label": label,
                "refresh_rate": refresh,
                "is_primary": is_primary,
                "width": scr.width,
                "height": scr.height,
            })

    except Exception as exc:
        print(f"WARNING: _detect_monitors() pyglet detection failed: {exc!r}")
        import traceback
        traceback.print_exc()

    if not monitors:
        monitors = [{"index": 0, "label": "显示器 1 (主屏)", "refresh_rate": 60, "is_primary": True}]

    return monitors


def _check_lsl_streams(stream_queries: Sequence[str], timeout: float = 2.0) -> dict[str, Any]:
    """检查所需的 LSL 流是否可用。

    按流 NAME 解析以进行快速、明确的匹配。
    OpenBCI GUI 默认值：obci_eeg1 (EEG), obci_eeg2 (Marker)。

    Args:
        stream_queries: 名称查询列表（例如 ['name="obci_eeg1"', 'name="obci_eeg2"']）
        timeout: LSL 解析超时时间（秒）

    Returns:
        包含以下键的字典：
        - 'all_found': bool，如果所有流都找到则为 True
        - 'results': 每个查询的 dict 列表，包含 'query', 'found', 'stream_info'
        - 'message': str，人类可读的状态消息
    """
    if pylsl is None:
        return {
            "all_found": False,
            "results": [],
            "message": "pylsl 未安装，无法检测 LSL 流",
        }

    import re

    results: list[dict[str, Any]] = []
    all_found = True

    try:
        for query in stream_queries:
            found = False
            matched_info: dict[str, Any] = {}

            # 解析查询：支持 'name="obci_eeg1"' 或 'type="EEG"'
            if "name=" in query:
                match = re.search(r'name="([^"]+)"', query)
                if match:
                    expected_name = match.group(1)
                    streams = pylsl.resolve_byprop("name", expected_name, timeout=timeout, minimum=1)
                    if streams:
                        found = True
                        info = streams[0]
                        matched_info = {
                            "name": info.name(),
                            "type": info.type(),
                            "channel_count": info.channel_count(),
                            "nominal_srate": info.nominal_srate(),
                        }
            elif "type=" in query:
                match = re.search(r'type="([^"]+)"', query)
                if match:
                    expected_type = match.group(1)
                    streams = pylsl.resolve_byprop("type", expected_type, timeout=timeout, minimum=1)
                    if streams:
                        found = True
                        info = streams[0]
                        matched_info = {
                            "name": info.name(),
                            "type": info.type(),
                            "channel_count": info.channel_count(),
                            "nominal_srate": info.nominal_srate(),
                        }

            if not found:
                all_found = False

            results.append({
                "query": query,
                "found": found,
                "stream_info": matched_info,
            })

    except Exception as exc:
        return {
            "all_found": False,
            "results": results,
            "message": f"LSL 检测出错: {exc}",
        }

    # 构建人类可读消息
    if all_found:
        message = f"[OK] LSL streams detected ({len(results)} streams)"
    else:
        missing = [r["query"] for r in results if not r["found"]]
        message = f"[!] LSL missing: {', '.join(missing)}"

    return {
        "all_found": all_found,
        "results": results,
        "message": message,
    }


def _load_dialog_defaults(args: argparse.Namespace) -> dict[str, Any]:
    """加载对话框默认值：YAML 配置 + 上次 session 文件 + CLI 参数。

    优先级：CLI 显式覆盖 > 保存的上次 session > YAML 默认值。
    """
    # 1. YAML 默认值
    script_dir = Path(__file__).resolve().parent
    yaml_dict = load_yaml_config(script_dir / "config_default.yaml")
    flat = _flatten_yaml(yaml_dict)
    defaults: dict[str, Any] = {
        "study_root": flat.get("labrecorder_study_root", ""),
        "recording_format": flat.get("labrecorder_recording_format", "xdf"),
        "csv_marker_mode": flat.get("labrecorder_csv_marker_mode", "legacy"),
        "participant": args.participant,
        "session": args.session,
        "run": args.run,
        "trial_mode": args.trial_mode,
        "mode": args.mode,
        "class_mode": args.class_mode,
        "display_index": args.display_index,
        "refresh_rate": flat.get("refresh_rate", 60),
        "fullscreen": flat.get("fullscreen", True),
        "blocks": flat.get("blocks", 2),
        "repeats_per_class": flat.get("repeats_per_class", 10),
        # SSVEP (mi_ssvep / pure_ssvep)
        "ssvep_flicker_mode": flat.get("ssvep_flicker_mode", "image"),
        "ssvep_waveform": flat.get("ssvep_waveform", "square"),
        "ssvep_display_mode": flat.get("ssvep_display_mode", "single_side"),
        "ssvep_left_freq": flat.get("ssvep_left_freq", 10.0),
        "ssvep_right_freq": flat.get("ssvep_right_freq", 15.0),
        # P300
        "p300_flicker_mode": flat.get("p300_flicker_mode", "image"),
        "p300_target_probability": flat.get("p300_target_probability", 0.25),
        # SSVEP Arousal
        "ssvep_arousal_freq_mode": flat.get("ssvep_arousal_freq_mode", "fixed"),
        "ssvep_arousal_fixed_freq_hz": flat.get("ssvep_arousal_fixed_freq_hz", 12.0),
        "ssvep_arousal_freq_min_hz": flat.get("ssvep_arousal_freq_min_hz", 8.0),
        "ssvep_arousal_freq_max_hz": flat.get("ssvep_arousal_freq_max_hz", 15.0),
        "ssvep_arousal_waveform": flat.get("ssvep_arousal_waveform", "sine"),
        "ssvep_arousal_stimulus_size": flat.get("ssvep_arousal_stimulus_size", 0.34),
        "ssvep_arousal_dim_opacity": flat.get("ssvep_arousal_dim_opacity", 0.0),
        # SSVEP Serial
        "ssvep_serial_cue_ssvep_freq_left_hz": flat.get("ssvep_serial_cue_ssvep_freq_left_hz", 10.0),
        "ssvep_serial_cue_ssvep_freq_right_hz": flat.get("ssvep_serial_cue_ssvep_freq_right_hz", 15.0),
        "ssvep_serial_cue_ssvep_mode": flat.get("ssvep_serial_cue_ssvep_mode", "freq_coded"),
        "ssvep_serial_same_freq_hz": flat.get("ssvep_serial_same_freq_hz", 12.0),
        "ssvep_serial_cue_ssvep_duration_s": flat.get("ssvep_serial_cue_ssvep_duration_s", 3.0),
        "ssvep_serial_gap_duration_s": flat.get("ssvep_serial_gap_duration_s", 1.0),
        "ssvep_serial_mi_duration_s": flat.get("ssvep_serial_mi_duration_s", 4.0),
        "ssvep_serial_waveform": flat.get("ssvep_serial_waveform", "sine"),
        "ssvep_serial_display_mode": flat.get("ssvep_serial_display_mode", "single_center"),
        "ssvep_serial_stimulus_width": flat.get("ssvep_serial_stimulus_width", 0.34),
        "ssvep_serial_stimulus_height": flat.get("ssvep_serial_stimulus_height", 0.34),
        "ssvep_serial_border_width": flat.get("ssvep_serial_border_width", 4.0),
        "ssvep_serial_dim_opacity": flat.get("ssvep_serial_dim_opacity", 0.0),
        # SSVEP RT (mi_ssvep_rt)
        "ssvep_rt_mi_enabled": flat.get("ssvep_rt_mi_enabled", False),
        "ssvep_rt_mi_checkpoint_path": flat.get("ssvep_rt_mi_checkpoint_path", "") or SSVEPRTConfig.mi_checkpoint_path,
        "ssvep_rt_classifier_window_s": flat.get("ssvep_rt_classifier_window_s", 1.5),
        "ssvep_rt_classifier_stride_s": flat.get("ssvep_rt_classifier_stride_s", 0.25),
        "ssvep_rt_confidence_threshold": flat.get("ssvep_rt_confidence_threshold", 0.15),
        "ssvep_rt_left_freq_hz": flat.get("ssvep_rt_left_freq_hz", 10.0),
        "ssvep_rt_right_freq_hz": flat.get("ssvep_rt_right_freq_hz", 15.0),
        "ssvep_rt_flicker_mode": flat.get("ssvep_rt_flicker_mode", "border"),
        "ssvep_rt_waveform": flat.get("ssvep_rt_waveform", "square"),
        "ssvep_rt_display_mode": flat.get("ssvep_rt_display_mode", "single_side"),
        "ssvep_rt_enable_diag": flat.get("ssvep_rt_enable_diag", False),
    }
    # 2. 用保存的值覆盖
    if _LAST_SESSION_FILE.exists():
        try:
            saved = json.loads(_LAST_SESSION_FILE.read_text(encoding="utf-8"))
            for key in defaults:
                if key in saved:
                    defaults[key] = saved[key]
        except (json.JSONDecodeError, OSError):
            pass
    # 3. CLI 显式覆盖（来自 CLI 的非 None 值）
    if args.study_root is not None:
        defaults["study_root"] = args.study_root
    if args.refresh_rate is not None:
        defaults["refresh_rate"] = args.refresh_rate
    return defaults


def _save_last_session(**kwargs: Any) -> None:
    """保存对话框值用于下次 session 预填充。"""
    try:
        _LAST_SESSION_FILE.write_text(
            json.dumps(kwargs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass  # 非关键的，静默失败


# tkinter 对话框：通用设置 + 范式专属面板根据 trial_mode 显隐切换
def show_session_dialog(args: argparse.Namespace,
                        defaults: dict[str, Any]) -> argparse.Namespace | None:
    """显示实验开始前的 session 参数 tkinter 对话框。

    从 defaults（YAML + 上次 session + CLI 覆盖）预填充字段。
    返回（可能修改后的）Namespace，如果用户取消则返回 None。
    """
    result: dict[str, Any] = {}
    cancelled = [False]

    # 检测显示器
    monitors = _detect_monitors()

    root = tk.Tk()
    root.title("MI 实验参数设置")
    root.resizable(True, True)  # 允许调整大小

    # 居中于屏幕
    root.update_idletasks()
    w, h = 580, 720
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")

    main_frame = ttk.Frame(root, padding=15)
    main_frame.pack(fill="both", expand=True)

    row = 0
    entries: dict[str, ttk.Entry] = {}

    # ── 数据根目录与浏览按钮 ──
    ttk.Label(main_frame, text="数据根目录").grid(row=row, column=0, sticky="w", pady=4)
    study_root_var = tk.StringVar(value=str(defaults.get("study_root", "")))
    study_root_entry = ttk.Entry(main_frame, textvariable=study_root_var, width=30)
    study_root_entry.grid(row=row, column=1, sticky="ew", pady=4, padx=(10, 0))

    def _browse_study_root() -> None:
        chosen = filedialog.askdirectory(
            initialdir=study_root_var.get() or None,
            title="选择数据根目录",
        )
        if chosen:
            study_root_var.set(chosen.replace("\\", "/"))

    browse_btn = ttk.Button(main_frame, text="浏览", command=_browse_study_root, width=5)
    browse_btn.grid(row=row, column=2, padx=(5, 0), pady=4)
    row += 1

    # ── 录制格式下拉框 ──
    ttk.Label(main_frame, text="录制格式").grid(row=row, column=0, sticky="w", pady=4)
    recording_format_var = tk.StringVar(value=str(defaults.get("recording_format", "xdf")))
    recording_format_combo = ttk.Combobox(
        main_frame, textvariable=recording_format_var,
        values=["xdf", "csv", "both"], state="readonly", width=10,
    )
    recording_format_combo.grid(row=row, column=1, sticky="w", pady=4, padx=(10, 0))
    recording_format_hint = ttk.Label(main_frame, text="(xdf=仅XDF, csv=仅CSV, both=同时录制)")
    recording_format_hint.grid(row=row, column=2, sticky="w", pady=4, padx=(5, 0))
    row += 1

    # ── CSV marker 模式下拉框（仅 format 为 csv 或 both 时可见）──
    csv_marker_mode_label = ttk.Label(main_frame, text="CSV Marker 模式")
    csv_marker_mode_label.grid(row=row, column=0, sticky="w", pady=4)
    csv_marker_mode_var = tk.StringVar(value=str(defaults.get("csv_marker_mode", "legacy")))
    csv_marker_mode_combo = ttk.Combobox(
        main_frame, textvariable=csv_marker_mode_var,
        values=["legacy", "detailed"], state="readonly", width=10,
    )
    csv_marker_mode_combo.grid(row=row, column=1, sticky="w", pady=4, padx=(10, 0))
    csv_marker_mode_hint = ttk.Label(main_frame, text="(legacy: 0/1/2, detailed: 实际值)")
    csv_marker_mode_hint.grid(row=row, column=2, sticky="w", pady=4, padx=(5, 0))
    row += 1

    # 根据录制格式切换的部件
    _csv_marker_widgets = [csv_marker_mode_label, csv_marker_mode_combo, csv_marker_mode_hint]

    def _toggle_csv_marker_mode(*_args: Any) -> None:
        show = recording_format_var.get() in ("csv", "both")
        for w in _csv_marker_widgets:
            if show:
                w.grid()
            else:
                w.grid_remove()

    recording_format_var.trace_add("write", _toggle_csv_marker_mode)
    _toggle_csv_marker_mode()  # 设置初始状态

    # ── 文本字段 ──
    fields: list[tuple[str, str, str]] = [
        ("participant", "被试编号 (Participant)", str(defaults.get("participant", "P001"))),
        ("session", "Session 编号", str(defaults.get("session", "S001"))),
        ("run", "Run 编号", str(defaults.get("run", 1))),
    ]

    # 保持 StringVar 引用活着以防止 GC 清除 Entry 值
    vars_: dict[str, tk.StringVar] = {}

    for key, label_text, default in fields:
        ttk.Label(main_frame, text=label_text).grid(row=row, column=0, sticky="w", pady=4)
        sv = tk.StringVar(value=default)
        vars_[key] = sv
        entry = ttk.Entry(main_frame, textvariable=sv, width=30)
        entry.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4, padx=(10, 0))
        entries[key] = entry
        row += 1

    # ── Trial mode 下拉框 ──
    ttk.Label(main_frame, text="实验范式 (Trial Mode)").grid(row=row, column=0, sticky="w", pady=4)
    trial_modes = ["pure_mi", "ao_mi", "mi_ssvep", "pure_ssvep", "mi_p300", "mixed", "mi_arrow", "mi_ssvep_arousal", "mi_ssvep_serial", "mi_ssvep_rt"]
    trial_var = tk.StringVar(value=str(defaults.get("trial_mode", "pure_mi")))
    trial_combo = ttk.Combobox(main_frame, textvariable=trial_var, values=trial_modes,
                               state="readonly", width=27)
    trial_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4, padx=(10, 0))
    row += 1

    # ── Mode 下拉框 ──
    ttk.Label(main_frame, text="模式 (Mode)").grid(row=row, column=0, sticky="w", pady=4)
    modes = ["pilot", "main", "custom"]
    mode_var = tk.StringVar(value=str(defaults.get("mode", "pilot")))
    mode_combo = ttk.Combobox(main_frame, textvariable=mode_var, values=modes,
                              state="readonly", width=27)
    mode_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4, padx=(10, 0))
    row += 1

    # ── 自定义模式设置（仅 mode == custom 时可见）──
    custom_frame = ttk.LabelFrame(main_frame, text="自定义模式设置", padding=8)
    custom_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
    custom_frame_row = row  # 记住实际网格行以便重新显示

    ttk.Label(custom_frame, text="Block 数量").grid(row=0, column=0, sticky="w", pady=2)
    custom_blocks_var = tk.StringVar(value=str(defaults.get("blocks", 2)))
    custom_blocks_entry = ttk.Entry(custom_frame, textvariable=custom_blocks_var, width=8)
    custom_blocks_entry.grid(row=0, column=1, sticky="w", pady=2, padx=(10, 0))

    ttk.Label(custom_frame, text="每类 Trial 数").grid(row=0, column=2, sticky="w", pady=2, padx=(20, 0))
    custom_trials_var = tk.StringVar(value=str(defaults.get("repeats_per_class", 10)))
    custom_trials_entry = ttk.Entry(custom_frame, textvariable=custom_trials_var, width=8)
    custom_trials_entry.grid(row=0, column=3, sticky="w", pady=2, padx=(10, 0))

    row += 1

    # ── 全屏复选框 ──
    fullscreen_var = tk.BooleanVar(value=bool(defaults.get("fullscreen", True)))
    fullscreen_check = ttk.Checkbutton(main_frame, text="全屏模式 (推荐，确保刷新率正确)", variable=fullscreen_var)
    fullscreen_check.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
    row += 1

    # ── 分类模式下拉框 ──
    ttk.Label(main_frame, text="分类模式 (Class Mode)").grid(row=row, column=0, sticky="w", pady=4)
    class_modes = ["binary", "ternary"]
    class_var = tk.StringVar(value=str(defaults.get("class_mode", "binary")))
    class_combo = ttk.Combobox(main_frame, textvariable=class_var, values=class_modes,
                               state="readonly", width=27)
    class_combo.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4, padx=(10, 0))
    row += 1

    # ── 显示器 + 刷新率（同一行）──
    ttk.Label(main_frame, text="显示器").grid(row=row, column=0, sticky="w", pady=4)
    monitor_labels = [m["label"] for m in monitors]
    # 查找默认 display 索引
    default_display = int(defaults.get("display_index", 0))
    if default_display >= len(monitor_labels):
        default_display = 0
    monitor_var = tk.StringVar(value=monitor_labels[default_display])
    monitor_combo = ttk.Combobox(main_frame, textvariable=monitor_var, values=monitor_labels,
                                 state="readonly", width=17)
    monitor_combo.grid(row=row, column=1, sticky="w", pady=4, padx=(10, 0))

    # 刷新率输入
    refresh_frame = ttk.Frame(main_frame)
    refresh_frame.grid(row=row, column=2, sticky="w", pady=4)
    default_refresh = monitors[default_display]["refresh_rate"]
    # 如果保存的刷新率与所选显示器匹配则使用，否则使用检测到的
    saved_refresh = defaults.get("refresh_rate", default_refresh)
    refresh_var = tk.StringVar(value=str(int(saved_refresh)))
    refresh_entry = ttk.Entry(refresh_frame, textvariable=refresh_var, width=5)
    refresh_entry.pack(side="left")
    ttk.Label(refresh_frame, text="Hz").pack(side="left", padx=(3, 0))

    def _on_monitor_change(_event: object = None) -> None:
        idx = monitor_labels.index(monitor_var.get()) if monitor_var.get() in monitor_labels else 0
        refresh_var.set(str(monitors[idx]["refresh_rate"]))

    monitor_combo.bind("<<ComboboxSelected>>", _on_monitor_change)
    row += 1

    # ── SSVEP 特定选项（仅 trial_mode == mi_ssvep 时可见）──
    ssvep_frame = ttk.LabelFrame(main_frame, text="SSVEP 设置", padding=8)
    ssvep_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
    row += 1

    # 闪烁模式
    ttk.Label(ssvep_frame, text="闪烁模式").grid(row=0, column=0, sticky="w", pady=2)
    flicker_mode_var = tk.StringVar(value=str(defaults.get("ssvep_flicker_mode", "image")))
    flicker_mode_combo = ttk.Combobox(ssvep_frame, textvariable=flicker_mode_var,
                                      values=["image", "border"], state="readonly", width=10)
    flicker_mode_combo.grid(row=0, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_frame, text="(image=图片透明度闪烁, border=边框颜色闪烁)").grid(
        row=0, column=2, sticky="w", pady=2, padx=(5, 0))

    # 波形
    ttk.Label(ssvep_frame, text="闪烁波形").grid(row=1, column=0, sticky="w", pady=2)
    waveform_var = tk.StringVar(value=str(defaults.get("ssvep_waveform", "square")))
    waveform_combo = ttk.Combobox(ssvep_frame, textvariable=waveform_var,
                                  values=["square", "sine"], state="readonly", width=10)
    waveform_combo.grid(row=1, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_frame, text="(square=方波更强信号, sine=正弦更舒适)").grid(
        row=1, column=2, sticky="w", pady=2, padx=(5, 0))

    # 左手频率
    ttk.Label(ssvep_frame, text="左手频率").grid(row=2, column=0, sticky="w", pady=2)
    left_freq_var = tk.StringVar(value=str(defaults.get("ssvep_left_freq", 10.0)))
    left_freq_entry = ttk.Entry(ssvep_frame, textvariable=left_freq_var, width=8)
    left_freq_entry.grid(row=2, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_frame, text="Hz").grid(row=2, column=2, sticky="w", pady=2)

    # 右手频率
    ttk.Label(ssvep_frame, text="右手频率").grid(row=3, column=0, sticky="w", pady=2)
    right_freq_var = tk.StringVar(value=str(defaults.get("ssvep_right_freq", 15.0)))
    right_freq_entry = ttk.Entry(ssvep_frame, textvariable=right_freq_var, width=8)
    right_freq_entry.grid(row=3, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_frame, text="Hz").grid(row=3, column=2, sticky="w", pady=2)

    # 显示模式
    ttk.Label(ssvep_frame, text="显示模式").grid(row=4, column=0, sticky="w", pady=2)
    ssvep_display_mode_var = tk.StringVar(value=str(defaults.get("ssvep_display_mode", "single_side")))
    ssvep_display_mode_combo = ttk.Combobox(ssvep_frame, textvariable=ssvep_display_mode_var,
                                             values=["single_side", "both_sides", "single_center"],
                                             state="readonly", width=12)
    ssvep_display_mode_combo.grid(row=4, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_frame, text="(single_side=仅目标侧, both_sides=双侧, single_center=居中)").grid(
        row=4, column=2, sticky="w", pady=2, padx=(5, 0))

    # ── P300 特定选项（仅 trial_mode == mi_p300 时可见）──
    p300_frame = ttk.LabelFrame(main_frame, text="P300 设置", padding=8)
    p300_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
    row += 1

    # P300 闪烁模式
    ttk.Label(p300_frame, text="闪烁模式").grid(row=0, column=0, sticky="w", pady=2)
    p300_flicker_mode_var = tk.StringVar(value=str(defaults.get("p300_flicker_mode", "image")))
    p300_flicker_mode_combo = ttk.Combobox(p300_frame, textvariable=p300_flicker_mode_var,
                                            values=["image", "border"], state="readonly", width=10)
    p300_flicker_mode_combo.grid(row=0, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(p300_frame, text="(image=图片透明度闪烁, border=边框颜色闪烁)").grid(
        row=0, column=2, sticky="w", pady=2, padx=(5, 0))

    # P300 目标概率
    ttk.Label(p300_frame, text="目标概率").grid(row=1, column=0, sticky="w", pady=2)
    p300_target_prob_var = tk.StringVar(value=str(defaults.get("p300_target_probability", 0.25)))
    p300_target_prob_entry = ttk.Entry(p300_frame, textvariable=p300_target_prob_var, width=8)
    p300_target_prob_entry.grid(row=1, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(p300_frame, text="(e.g. 0.25 = 25% 目标侧闪烁，自动保证至少1次)").grid(
        row=1, column=2, sticky="w", pady=2, padx=(5, 0))

    # ── SSVEP Arousal 特定选项（仅 trial_mode == mi_ssvep_arousal 时可见）──
    ssvep_arousal_frame = ttk.LabelFrame(main_frame, text="SSVEP Arousal 设置", padding=8)
    ssvep_arousal_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
    row += 1

    # 频率模式
    ttk.Label(ssvep_arousal_frame, text="频率模式").grid(row=0, column=0, sticky="w", pady=2)
    ssvep_arousal_freq_mode_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_freq_mode", "fixed")))
    ssvep_arousal_freq_mode_combo = ttk.Combobox(ssvep_arousal_frame, textvariable=ssvep_arousal_freq_mode_var,
                                                  values=["fixed", "random"], state="readonly", width=10)
    ssvep_arousal_freq_mode_combo.grid(row=0, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_arousal_frame, text="(fixed=固定频率, random=随机频率)").grid(
        row=0, column=2, sticky="w", pady=2, padx=(5, 0))

    # 固定频率
    ssvep_arousal_fixed_freq_label = ttk.Label(ssvep_arousal_frame, text="固定频率")
    ssvep_arousal_fixed_freq_label.grid(row=1, column=0, sticky="w", pady=2)
    ssvep_arousal_fixed_freq_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_fixed_freq_hz", 20.0)))
    ssvep_arousal_fixed_freq_entry = ttk.Entry(ssvep_arousal_frame, textvariable=ssvep_arousal_fixed_freq_var, width=8)
    ssvep_arousal_fixed_freq_entry.grid(row=1, column=1, sticky="w", pady=2, padx=(10, 0))
    ssvep_arousal_fixed_freq_hz_label = ttk.Label(ssvep_arousal_frame, text="Hz")
    ssvep_arousal_fixed_freq_hz_label.grid(row=1, column=2, sticky="w", pady=2)

    # 频率最小值
    ssvep_arousal_freq_min_label = ttk.Label(ssvep_arousal_frame, text="最小频率")
    ssvep_arousal_freq_min_label.grid(row=2, column=0, sticky="w", pady=2)
    ssvep_arousal_freq_min_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_freq_min_hz", 18.0)))
    ssvep_arousal_freq_min_entry = ttk.Entry(ssvep_arousal_frame, textvariable=ssvep_arousal_freq_min_var, width=8)
    ssvep_arousal_freq_min_entry.grid(row=2, column=1, sticky="w", pady=2, padx=(10, 0))
    ssvep_arousal_freq_min_hz_label = ttk.Label(ssvep_arousal_frame, text="Hz")
    ssvep_arousal_freq_min_hz_label.grid(row=2, column=2, sticky="w", pady=2)

    # 频率最大值
    ssvep_arousal_freq_max_label = ttk.Label(ssvep_arousal_frame, text="最大频率")
    ssvep_arousal_freq_max_label.grid(row=3, column=0, sticky="w", pady=2)
    ssvep_arousal_freq_max_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_freq_max_hz", 25.0)))
    ssvep_arousal_freq_max_entry = ttk.Entry(ssvep_arousal_frame, textvariable=ssvep_arousal_freq_max_var, width=8)
    ssvep_arousal_freq_max_entry.grid(row=3, column=1, sticky="w", pady=2, padx=(10, 0))
    ssvep_arousal_freq_max_hz_label = ttk.Label(ssvep_arousal_frame, text="Hz")
    ssvep_arousal_freq_max_hz_label.grid(row=3, column=2, sticky="w", pady=2)

    # 波形
    ttk.Label(ssvep_arousal_frame, text="闪烁波形").grid(row=4, column=0, sticky="w", pady=2)
    ssvep_arousal_waveform_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_waveform", "sine")))
    ssvep_arousal_waveform_combo = ttk.Combobox(ssvep_arousal_frame, textvariable=ssvep_arousal_waveform_var,
                                                 values=["square", "sine"], state="readonly", width=10)
    ssvep_arousal_waveform_combo.grid(row=4, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_arousal_frame, text="(square=方波更强信号, sine=正弦更舒适)").grid(
        row=4, column=2, sticky="w", pady=2, padx=(5, 0))

    # 提示样式
    ttk.Label(ssvep_arousal_frame, text="提示样式").grid(row=5, column=0, sticky="w", pady=2)
    ssvep_arousal_cue_style_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_cue_style", "arrow")))
    ssvep_arousal_cue_style_combo = ttk.Combobox(ssvep_arousal_frame, textvariable=ssvep_arousal_cue_style_var,
                                                  values=["arrow", "image"], state="readonly", width=10)
    ssvep_arousal_cue_style_combo.grid(row=5, column=1, sticky="w", pady=2, padx=(10, 0))

    # 任务样式
    ttk.Label(ssvep_arousal_frame, text="任务样式").grid(row=6, column=0, sticky="w", pady=2)
    ssvep_arousal_task_style_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_task_style", "arrow")))
    ssvep_arousal_task_style_combo = ttk.Combobox(ssvep_arousal_frame, textvariable=ssvep_arousal_task_style_var,
                                                   values=["arrow", "image"], state="readonly", width=10)
    ssvep_arousal_task_style_combo.grid(row=6, column=1, sticky="w", pady=2, padx=(10, 0))

    # 刺激尺寸
    ttk.Label(ssvep_arousal_frame, text="刺激尺寸").grid(row=7, column=0, sticky="w", pady=2)
    ssvep_arousal_stimulus_size_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_stimulus_size", 0.35)))
    ssvep_arousal_stimulus_size_entry = ttk.Entry(ssvep_arousal_frame, textvariable=ssvep_arousal_stimulus_size_var, width=8)
    ssvep_arousal_stimulus_size_entry.grid(row=7, column=1, sticky="w", pady=2, padx=(10, 0))

    # 透明度
    ttk.Label(ssvep_arousal_frame, text="透明度").grid(row=8, column=0, sticky="w", pady=2)
    ssvep_arousal_dim_opacity_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_dim_opacity", 0.0)))
    ssvep_arousal_dim_opacity_entry = ttk.Entry(ssvep_arousal_frame, textvariable=ssvep_arousal_dim_opacity_var, width=8)
    ssvep_arousal_dim_opacity_entry.grid(row=8, column=1, sticky="w", pady=2, padx=(10, 0))

    # 箭头颜色
    ttk.Label(ssvep_arousal_frame, text="箭头颜色").grid(row=9, column=0, sticky="w", pady=2)
    ssvep_arousal_arrow_color_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_arrow_color", "white")))
    ssvep_arousal_arrow_color_entry = ttk.Entry(ssvep_arousal_frame, textvariable=ssvep_arousal_arrow_color_var, width=8)
    ssvep_arousal_arrow_color_entry.grid(row=9, column=1, sticky="w", pady=2, padx=(10, 0))

    # 箭头大小
    ttk.Label(ssvep_arousal_frame, text="箭头大小").grid(row=10, column=0, sticky="w", pady=2)
    ssvep_arousal_arrow_height_var = tk.StringVar(value=str(defaults.get("ssvep_arousal_arrow_height", 0.20)))
    ssvep_arousal_arrow_height_entry = ttk.Entry(ssvep_arousal_frame, textvariable=ssvep_arousal_arrow_height_var, width=8)
    ssvep_arousal_arrow_height_entry.grid(row=10, column=1, sticky="w", pady=2, padx=(10, 0))

    # ── SSVEP Serial 特定选项（仅 trial_mode == mi_ssvep_serial 时可见）──
    ssvep_serial_frame = ttk.LabelFrame(main_frame, text="SSVEP Serial 设置", padding=8)
    ssvep_serial_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
    row += 1

    # 提示模式
    ttk.Label(ssvep_serial_frame, text="提示模式").grid(row=0, column=0, sticky="w", pady=2)
    ssvep_serial_cue_ssvep_mode_var = tk.StringVar(value=str(defaults.get("ssvep_serial_cue_ssvep_mode", "frequency_coded")))
    ssvep_serial_cue_ssvep_mode_combo = ttk.Combobox(ssvep_serial_frame, textvariable=ssvep_serial_cue_ssvep_mode_var,
                                                   values=["frequency_coded", "same_freq"], state="readonly", width=15)
    ssvep_serial_cue_ssvep_mode_combo.grid(row=0, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_serial_frame, text="(frequency_coded=频率编码方向, same_freq=同频不编码方向)").grid(
        row=0, column=2, sticky="w", pady=2, padx=(5, 0))

    # 提示左频率
    ssvep_serial_cue_ssvep_freq_left_label = ttk.Label(ssvep_serial_frame, text="提示左频率")
    ssvep_serial_cue_ssvep_freq_left_label.grid(row=1, column=0, sticky="w", pady=2)
    ssvep_serial_cue_ssvep_freq_left_var = tk.StringVar(value=str(defaults.get("ssvep_serial_cue_ssvep_freq_left_hz", 10.0)))
    ssvep_serial_cue_ssvep_freq_left_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_cue_ssvep_freq_left_var, width=8)
    ssvep_serial_cue_ssvep_freq_left_entry.grid(row=1, column=1, sticky="w", pady=2, padx=(10, 0))
    ssvep_serial_cue_ssvep_freq_left_hz_label = ttk.Label(ssvep_serial_frame, text="Hz")
    ssvep_serial_cue_ssvep_freq_left_hz_label.grid(row=1, column=2, sticky="w", pady=2)

    # 提示右频率
    ssvep_serial_cue_ssvep_freq_right_label = ttk.Label(ssvep_serial_frame, text="提示右频率")
    ssvep_serial_cue_ssvep_freq_right_label.grid(row=2, column=0, sticky="w", pady=2)
    ssvep_serial_cue_ssvep_freq_right_var = tk.StringVar(value=str(defaults.get("ssvep_serial_cue_ssvep_freq_right_hz", 15.0)))
    ssvep_serial_cue_ssvep_freq_right_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_cue_ssvep_freq_right_var, width=8)
    ssvep_serial_cue_ssvep_freq_right_entry.grid(row=2, column=1, sticky="w", pady=2, padx=(10, 0))
    ssvep_serial_cue_ssvep_freq_right_hz_label = ttk.Label(ssvep_serial_frame, text="Hz")
    ssvep_serial_cue_ssvep_freq_right_hz_label.grid(row=2, column=2, sticky="w", pady=2)

    # 同频模式频率
    ssvep_serial_same_freq_label = ttk.Label(ssvep_serial_frame, text="同频模式频率")
    ssvep_serial_same_freq_label.grid(row=3, column=0, sticky="w", pady=2)
    ssvep_serial_same_freq_var = tk.StringVar(value=str(defaults.get("ssvep_serial_same_freq_hz", 20.0)))
    ssvep_serial_same_freq_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_same_freq_var, width=8)
    ssvep_serial_same_freq_entry.grid(row=3, column=1, sticky="w", pady=2, padx=(10, 0))
    ssvep_serial_same_freq_hz_label = ttk.Label(ssvep_serial_frame, text="Hz")
    ssvep_serial_same_freq_hz_label.grid(row=3, column=2, sticky="w", pady=2)

    # 提示时长
    ttk.Label(ssvep_serial_frame, text="提示时长").grid(row=4, column=0, sticky="w", pady=2)
    ssvep_serial_cue_ssvep_duration_var = tk.StringVar(value=str(defaults.get("ssvep_serial_cue_ssvep_duration_s", 2.0)))
    ssvep_serial_cue_ssvep_duration_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_cue_ssvep_duration_var, width=8)
    ssvep_serial_cue_ssvep_duration_entry.grid(row=4, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_serial_frame, text="s").grid(row=4, column=2, sticky="w", pady=2)

    # 间隔时长
    ttk.Label(ssvep_serial_frame, text="间隔时长").grid(row=5, column=0, sticky="w", pady=2)
    ssvep_serial_gap_duration_var = tk.StringVar(value=str(defaults.get("ssvep_serial_gap_duration_s", 2.0)))
    ssvep_serial_gap_duration_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_gap_duration_var, width=8)
    ssvep_serial_gap_duration_entry.grid(row=5, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_serial_frame, text="s").grid(row=5, column=2, sticky="w", pady=2)

    # MI 时长
    ttk.Label(ssvep_serial_frame, text="MI时长").grid(row=6, column=0, sticky="w", pady=2)
    ssvep_serial_mi_duration_var = tk.StringVar(value=str(defaults.get("ssvep_serial_mi_duration_s", 4.0)))
    ssvep_serial_mi_duration_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_mi_duration_var, width=8)
    ssvep_serial_mi_duration_entry.grid(row=6, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_serial_frame, text="s").grid(row=6, column=2, sticky="w", pady=2)

    # 波形
    ttk.Label(ssvep_serial_frame, text="闪烁波形").grid(row=7, column=0, sticky="w", pady=2)
    ssvep_serial_waveform_var = tk.StringVar(value=str(defaults.get("ssvep_serial_waveform", "sine")))
    ssvep_serial_waveform_combo = ttk.Combobox(ssvep_serial_frame, textvariable=ssvep_serial_waveform_var,
                                              values=["square", "sine"], state="readonly", width=10)
    ssvep_serial_waveform_combo.grid(row=7, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_serial_frame, text="(square=方波更强信号, sine=正弦更舒适)").grid(
        row=7, column=2, sticky="w", pady=2, padx=(5, 0))

    # 提示样式
    ttk.Label(ssvep_serial_frame, text="提示样式").grid(row=8, column=0, sticky="w", pady=2)
    ssvep_serial_cue_style_var = tk.StringVar(value=str(defaults.get("ssvep_serial_cue_style", "arrow")))
    ssvep_serial_cue_style_combo = ttk.Combobox(ssvep_serial_frame, textvariable=ssvep_serial_cue_style_var,
                                              values=["arrow", "image"], state="readonly", width=10)
    ssvep_serial_cue_style_combo.grid(row=8, column=1, sticky="w", pady=2, padx=(10, 0))

    # 任务样式
    ttk.Label(ssvep_serial_frame, text="任务样式").grid(row=9, column=0, sticky="w", pady=2)
    ssvep_serial_task_style_var = tk.StringVar(value=str(defaults.get("ssvep_serial_task_style", "arrow")))
    ssvep_serial_task_style_combo = ttk.Combobox(ssvep_serial_frame, textvariable=ssvep_serial_task_style_var,
                                               values=["arrow", "image"], state="readonly", width=10)
    ssvep_serial_task_style_combo.grid(row=9, column=1, sticky="w", pady=2, padx=(10, 0))

    # 显示模式
    ttk.Label(ssvep_serial_frame, text="显示模式").grid(row=10, column=0, sticky="w", pady=2)
    ssvep_serial_display_mode_var = tk.StringVar(value=str(defaults.get("ssvep_serial_display_mode", "single_center")))
    ssvep_serial_display_mode_combo = ttk.Combobox(ssvep_serial_frame, textvariable=ssvep_serial_display_mode_var,
                                                values=["single_center", "both_sides"], state="readonly", width=12)
    ssvep_serial_display_mode_combo.grid(row=10, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_serial_frame, text="(single_center=仅目标侧居中, both_sides=双侧)").grid(
        row=10, column=2, sticky="w", pady=2, padx=(5, 0))

    # 刺激宽度
    ttk.Label(ssvep_serial_frame, text="刺激宽度").grid(row=11, column=0, sticky="w", pady=2)
    ssvep_serial_stimulus_width_var = tk.StringVar(value=str(defaults.get("ssvep_serial_stimulus_width", 0.35)))
    ssvep_serial_stimulus_width_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_stimulus_width_var, width=8)
    ssvep_serial_stimulus_width_entry.grid(row=11, column=1, sticky="w", pady=2, padx=(10, 0))

    # 刺激高度
    ttk.Label(ssvep_serial_frame, text="刺激高度").grid(row=12, column=0, sticky="w", pady=2)
    ssvep_serial_stimulus_height_var = tk.StringVar(value=str(defaults.get("ssvep_serial_stimulus_height", 0.35)))
    ssvep_serial_stimulus_height_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_stimulus_height_var, width=8)
    ssvep_serial_stimulus_height_entry.grid(row=12, column=1, sticky="w", pady=2, padx=(10, 0))

    # 边框宽度
    ttk.Label(ssvep_serial_frame, text="边框宽度").grid(row=13, column=0, sticky="w", pady=2)
    ssvep_serial_border_width_var = tk.StringVar(value=str(defaults.get("ssvep_serial_border_width", 4.0)))
    ssvep_serial_border_width_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_border_width_var, width=8)
    ssvep_serial_border_width_entry.grid(row=13, column=1, sticky="w", pady=2, padx=(10, 0))

    # 透明度
    ttk.Label(ssvep_serial_frame, text="透明度").grid(row=14, column=0, sticky="w", pady=2)
    ssvep_serial_dim_opacity_var = tk.StringVar(value=str(defaults.get("ssvep_serial_dim_opacity", 0.0)))
    ssvep_serial_dim_opacity_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_dim_opacity_var, width=8)
    ssvep_serial_dim_opacity_entry.grid(row=14, column=1, sticky="w", pady=2, padx=(10, 0))

    # 箭头颜色
    ttk.Label(ssvep_serial_frame, text="箭头颜色").grid(row=15, column=0, sticky="w", pady=2)
    ssvep_serial_arrow_color_var = tk.StringVar(value=str(defaults.get("ssvep_serial_arrow_color", "white")))
    ssvep_serial_arrow_color_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_arrow_color_var, width=8)
    ssvep_serial_arrow_color_entry.grid(row=15, column=1, sticky="w", pady=2, padx=(10, 0))

    # 箭头大小
    ttk.Label(ssvep_serial_frame, text="箭头大小").grid(row=16, column=0, sticky="w", pady=2)
    ssvep_serial_arrow_height_var = tk.StringVar(value=str(defaults.get("ssvep_serial_arrow_height", 0.20)))
    ssvep_serial_arrow_height_entry = ttk.Entry(ssvep_serial_frame, textvariable=ssvep_serial_arrow_height_var, width=8)
    ssvep_serial_arrow_height_entry.grid(row=16, column=1, sticky="w", pady=2, padx=(10, 0))

    # ── SSVEP RT 特定选项（仅 trial_mode == mi_ssvep_rt 时可见）──
    ssvep_rt_frame = ttk.LabelFrame(main_frame, text="MI+SSVEP 实时分类设置", padding=8)
    ssvep_rt_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
    row += 1

    # MI 启用复选框
    _rt_mi_enabled_default = str(defaults.get("ssvep_rt_mi_enabled", "False")).lower() in ("true", "1", "yes", "on")
    ssvep_rt_mi_enabled_var = tk.BooleanVar(value=_rt_mi_enabled_default)

    def _on_rt_mi_toggle() -> None:
        if ssvep_rt_mi_enabled_var.get():
            _rt_mi_checkpoint_frame.grid()
        else:
            _rt_mi_checkpoint_frame.grid_remove()

    ttk.Checkbutton(
        ssvep_rt_frame, text="启用MI模型（不勾选仅用FBCCA）",
        variable=ssvep_rt_mi_enabled_var, command=_on_rt_mi_toggle,
    ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

    # MI 检查点路径 — 仅 MI 启用时可见
    _rt_mi_checkpoint_frame = ttk.Frame(ssvep_rt_frame)
    _rt_mi_checkpoint_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=2)
    _rt_mi_checkpoint_frame.columnconfigure(1, weight=1)

    ttk.Label(_rt_mi_checkpoint_frame, text="MI模型路径").grid(
        row=0, column=0, sticky="w", pady=0, padx=(0, 5))
    ssvep_rt_checkpoint_var = tk.StringVar(value=str(defaults.get("ssvep_rt_mi_checkpoint_path", "") or SSVEPRTConfig.mi_checkpoint_path))
    ssvep_rt_checkpoint_entry = ttk.Entry(_rt_mi_checkpoint_frame, textvariable=ssvep_rt_checkpoint_var, width=25)
    ssvep_rt_checkpoint_entry.grid(row=0, column=1, sticky="ew", pady=0, padx=(0, 5))

    def _browse_rt_checkpoint() -> None:
        chosen = filedialog.askopenfilename(
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")],
            initialdir=None,
        )
        if chosen:
            ssvep_rt_checkpoint_var.set(chosen)

    ttk.Button(_rt_mi_checkpoint_frame, text="浏览...", command=_browse_rt_checkpoint).grid(
        row=0, column=2, sticky="w", pady=0)

    ttk.Label(ssvep_rt_frame, text="（勾选后选择.pth模型文件；不勾选则仅用FBCCA）").grid(
        row=2, column=0, columnspan=3, sticky="w", pady=0)

    # 根据保存的默认值设置初始可见性
    if not _rt_mi_enabled_default:
        _rt_mi_checkpoint_frame.grid_remove()

    # 分类窗口大小
    ttk.Label(ssvep_rt_frame, text="分类窗口 (s)").grid(row=3, column=0, sticky="w", pady=2)
    ssvep_rt_window_var = tk.StringVar(value=str(defaults.get("ssvep_rt_classifier_window_s", 1.5)))
    ttk.Entry(ssvep_rt_frame, textvariable=ssvep_rt_window_var, width=8).grid(
        row=3, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_rt_frame, text="秒（默认1.5）").grid(row=3, column=2, sticky="w", pady=2)

    # 分类步长
    ttk.Label(ssvep_rt_frame, text="滑动步长 (s)").grid(row=4, column=0, sticky="w", pady=2)
    ssvep_rt_stride_var = tk.StringVar(value=str(defaults.get("ssvep_rt_classifier_stride_s", 0.25)))
    ttk.Entry(ssvep_rt_frame, textvariable=ssvep_rt_stride_var, width=8).grid(
        row=4, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_rt_frame, text="秒（默认0.25）").grid(row=4, column=2, sticky="w", pady=2)

    # 置信度阈值
    ttk.Label(ssvep_rt_frame, text="置信度阈值").grid(row=5, column=0, sticky="w", pady=2)
    ssvep_rt_conf_var = tk.StringVar(value=str(defaults.get("ssvep_rt_confidence_threshold", 0.15)))
    ttk.Entry(ssvep_rt_frame, textvariable=ssvep_rt_conf_var, width=8).grid(
        row=5, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_rt_frame, text="（默认0.15，基于相关系数差值）").grid(row=5, column=2, sticky="w", pady=2)

    # LSL 诊断模式
    _rt_diag_default = str(defaults.get("ssvep_rt_enable_diag", "False")).lower() in ("true", "1", "yes", "on")
    ssvep_rt_enable_diag_var = tk.BooleanVar(value=_rt_diag_default)
    ttk.Checkbutton(
        ssvep_rt_frame, text="启用LSL诊断（检测数据间隙/丢包，保存原始窗口）",
        variable=ssvep_rt_enable_diag_var,
    ).grid(row=6, column=0, columnspan=3, sticky="w", pady=2)

    # SSVEP 显示设置（复用 ssvep_frame 的模式）
    ttk.Label(ssvep_rt_frame, text="闪烁模式").grid(row=7, column=0, sticky="w", pady=2)
    ssvep_rt_flicker_mode_var = tk.StringVar(value=str(defaults.get("ssvep_rt_flicker_mode", "border")))
    ttk.Combobox(ssvep_rt_frame, textvariable=ssvep_rt_flicker_mode_var,
                 values=["image", "border"], state="readonly", width=10).grid(
        row=7, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_rt_frame, text="(image=图片透明度闪烁, border=边框颜色闪烁)").grid(
        row=7, column=2, sticky="w", pady=2, padx=(5, 0))

    ttk.Label(ssvep_rt_frame, text="闪烁波形").grid(row=8, column=0, sticky="w", pady=2)
    ssvep_rt_waveform_var = tk.StringVar(value=str(defaults.get("ssvep_rt_waveform", "square")))
    ttk.Combobox(ssvep_rt_frame, textvariable=ssvep_rt_waveform_var,
                 values=["square", "sine"], state="readonly", width=10).grid(
        row=8, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_rt_frame, text="(square=方波更强信号, sine=正弦更舒适)").grid(
        row=8, column=2, sticky="w", pady=2, padx=(5, 0))

    ttk.Label(ssvep_rt_frame, text="左手频率").grid(row=9, column=0, sticky="w", pady=2)
    ssvep_rt_left_freq_var = tk.StringVar(value=str(defaults.get("ssvep_rt_left_freq_hz", 10.0)))
    ttk.Entry(ssvep_rt_frame, textvariable=ssvep_rt_left_freq_var, width=8).grid(
        row=9, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_rt_frame, text="Hz").grid(row=9, column=2, sticky="w", pady=2)

    ttk.Label(ssvep_rt_frame, text="右手频率").grid(row=10, column=0, sticky="w", pady=2)
    ssvep_rt_right_freq_var = tk.StringVar(value=str(defaults.get("ssvep_rt_right_freq_hz", 15.0)))
    ttk.Entry(ssvep_rt_frame, textvariable=ssvep_rt_right_freq_var, width=8).grid(
        row=10, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_rt_frame, text="Hz").grid(row=10, column=2, sticky="w", pady=2)

    ttk.Label(ssvep_rt_frame, text="显示模式").grid(row=11, column=0, sticky="w", pady=2)
    ssvep_rt_display_mode_var = tk.StringVar(value=str(defaults.get("ssvep_rt_display_mode", "single_side")))
    ttk.Combobox(ssvep_rt_frame, textvariable=ssvep_rt_display_mode_var,
                 values=["single_side", "both_sides", "single_center"],
                 state="readonly", width=12).grid(
        row=11, column=1, sticky="w", pady=2, padx=(10, 0))
    ttk.Label(ssvep_rt_frame, text="(single_side=仅目标侧, both_sides=双侧, single_center=居中)").grid(
        row=11, column=2, sticky="w", pady=2, padx=(5, 0))

    # 根据 trial_mode 显示/隐藏 SSVEP/P300/SSVEP Arousal/SSVEP Serial/SSVEP RT 面板
    ssvep_frame_row = row - 5
    p300_frame_row = row - 4
    ssvep_arousal_frame_row = row - 3
    ssvep_serial_frame_row = row - 2
    ssvep_rt_frame_row = row - 1

    def _on_arousal_freq_mode_change(_event: object = None) -> None:
        """根据 SSVEP Arousal 频率模式选择显示/隐藏字段。"""
        is_fixed = ssvep_arousal_freq_mode_var.get() == "fixed"
        # 固定频率仅在 fixed 模式下可见
        if is_fixed:
            ssvep_arousal_fixed_freq_label.grid(row=1, column=0, sticky="w", pady=2)
            ssvep_arousal_fixed_freq_entry.grid(row=1, column=1, sticky="w", pady=2, padx=(10, 0))
            ssvep_arousal_fixed_freq_hz_label.grid(row=1, column=2, sticky="w", pady=2)
        else:
            ssvep_arousal_fixed_freq_label.grid_forget()
            ssvep_arousal_fixed_freq_entry.grid_forget()
            ssvep_arousal_fixed_freq_hz_label.grid_forget()
        # 最小/最大频率仅在 random 模式下可见
        if not is_fixed:
            ssvep_arousal_freq_min_label.grid(row=2, column=0, sticky="w", pady=2)
            ssvep_arousal_freq_min_entry.grid(row=2, column=1, sticky="w", pady=2, padx=(10, 0))
            ssvep_arousal_freq_min_hz_label.grid(row=2, column=2, sticky="w", pady=2)
            ssvep_arousal_freq_max_label.grid(row=3, column=0, sticky="w", pady=2)
            ssvep_arousal_freq_max_entry.grid(row=3, column=1, sticky="w", pady=2, padx=(10, 0))
            ssvep_arousal_freq_max_hz_label.grid(row=3, column=2, sticky="w", pady=2)
        else:
            ssvep_arousal_freq_min_label.grid_forget()
            ssvep_arousal_freq_min_entry.grid_forget()
            ssvep_arousal_freq_min_hz_label.grid_forget()
            ssvep_arousal_freq_max_label.grid_forget()
            ssvep_arousal_freq_max_entry.grid_forget()
            ssvep_arousal_freq_max_hz_label.grid_forget()

    def _on_serial_cue_mode_change(_event: object = None) -> None:
        """根据 SSVEP Serial 提示模式选择显示/隐藏字段。"""
        is_freq_coded = ssvep_serial_cue_ssvep_mode_var.get() == "frequency_coded"
        # 左右频率仅在 frequency_coded 模式下可见
        if is_freq_coded:
            ssvep_serial_cue_ssvep_freq_left_label.grid(row=1, column=0, sticky="w", pady=2)
            ssvep_serial_cue_ssvep_freq_left_entry.grid(row=1, column=1, sticky="w", pady=2, padx=(10, 0))
            ssvep_serial_cue_ssvep_freq_left_hz_label.grid(row=1, column=2, sticky="w", pady=2)
            ssvep_serial_cue_ssvep_freq_right_label.grid(row=2, column=0, sticky="w", pady=2)
            ssvep_serial_cue_ssvep_freq_right_entry.grid(row=2, column=1, sticky="w", pady=2, padx=(10, 0))
            ssvep_serial_cue_ssvep_freq_right_hz_label.grid(row=2, column=2, sticky="w", pady=2)
        else:
            ssvep_serial_cue_ssvep_freq_left_label.grid_forget()
            ssvep_serial_cue_ssvep_freq_left_entry.grid_forget()
            ssvep_serial_cue_ssvep_freq_left_hz_label.grid_forget()
            ssvep_serial_cue_ssvep_freq_right_label.grid_forget()
            ssvep_serial_cue_ssvep_freq_right_entry.grid_forget()
            ssvep_serial_cue_ssvep_freq_right_hz_label.grid_forget()
        # 同频仅在 same_freq 模式下可见
        if not is_freq_coded:
            ssvep_serial_same_freq_label.grid(row=3, column=0, sticky="w", pady=2)
            ssvep_serial_same_freq_entry.grid(row=3, column=1, sticky="w", pady=2, padx=(10, 0))
            ssvep_serial_same_freq_hz_label.grid(row=3, column=2, sticky="w", pady=2)
        else:
            ssvep_serial_same_freq_label.grid_forget()
            ssvep_serial_same_freq_entry.grid_forget()
            ssvep_serial_same_freq_hz_label.grid_forget()

    ssvep_arousal_freq_mode_combo.bind("<<ComboboxSelected>>", _on_arousal_freq_mode_change)
    ssvep_serial_cue_ssvep_mode_combo.bind("<<ComboboxSelected>>", _on_serial_cue_mode_change)
    # 应用初始可见性
    _on_arousal_freq_mode_change()
    _on_serial_cue_mode_change()

    def _on_trial_mode_change(_event: object = None) -> None:
        if trial_var.get() in ("mi_ssvep", "pure_ssvep"):
            ssvep_frame.grid(row=ssvep_frame_row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
            p300_frame.grid_forget()
            ssvep_arousal_frame.grid_forget()
            ssvep_serial_frame.grid_forget()
            ssvep_rt_frame.grid_forget()
        elif trial_var.get() == "mi_p300":
            ssvep_frame.grid_forget()
            p300_frame.grid(row=p300_frame_row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
            ssvep_arousal_frame.grid_forget()
            ssvep_serial_frame.grid_forget()
            ssvep_rt_frame.grid_forget()
        elif trial_var.get() == "mi_ssvep_arousal":
            ssvep_frame.grid_forget()
            p300_frame.grid_forget()
            ssvep_arousal_frame.grid(row=ssvep_arousal_frame_row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
            ssvep_serial_frame.grid_forget()
            ssvep_rt_frame.grid_forget()
        elif trial_var.get() == "mi_ssvep_serial":
            ssvep_frame.grid_forget()
            p300_frame.grid_forget()
            ssvep_arousal_frame.grid_forget()
            ssvep_serial_frame.grid(row=ssvep_serial_frame_row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
            ssvep_rt_frame.grid_forget()
        elif trial_var.get() == "mi_ssvep_rt":
            ssvep_frame.grid_forget()
            p300_frame.grid_forget()
            ssvep_arousal_frame.grid_forget()
            ssvep_serial_frame.grid_forget()
            ssvep_rt_frame.grid(row=ssvep_rt_frame_row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        else:
            ssvep_frame.grid_forget()
            p300_frame.grid_forget()
            ssvep_arousal_frame.grid_forget()
            ssvep_serial_frame.grid_forget()
            ssvep_rt_frame.grid_forget()

    def _on_mode_change(_event: object = None) -> None:
        if mode_var.get() == "custom":
            custom_frame.grid(row=custom_frame_row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        else:
            custom_frame.grid_forget()

    trial_combo.bind("<<ComboboxSelected>>", _on_trial_mode_change)
    mode_combo.bind("<<ComboboxSelected>>", _on_mode_change)

    # 初始可见性
    if trial_var.get() in ("mi_ssvep", "pure_ssvep"):
        p300_frame.grid_forget()
        ssvep_arousal_frame.grid_forget()
        ssvep_serial_frame.grid_forget()
        ssvep_rt_frame.grid_forget()
    elif trial_var.get() == "mi_p300":
        ssvep_frame.grid_forget()
        ssvep_arousal_frame.grid_forget()
        ssvep_serial_frame.grid_forget()
        ssvep_rt_frame.grid_forget()
    elif trial_var.get() == "mi_ssvep_arousal":
        ssvep_frame.grid_forget()
        p300_frame.grid_forget()
        ssvep_serial_frame.grid_forget()
        ssvep_rt_frame.grid_forget()
    elif trial_var.get() == "mi_ssvep_serial":
        ssvep_frame.grid_forget()
        p300_frame.grid_forget()
        ssvep_arousal_frame.grid_forget()
        ssvep_rt_frame.grid_forget()
    elif trial_var.get() == "mi_ssvep_rt":
        ssvep_frame.grid_forget()
        p300_frame.grid_forget()
        ssvep_arousal_frame.grid_forget()
        ssvep_serial_frame.grid_forget()
    else:
        ssvep_frame.grid_forget()
        p300_frame.grid_forget()
        ssvep_arousal_frame.grid_forget()
        ssvep_serial_frame.grid_forget()
        ssvep_rt_frame.grid_forget()

    if mode_var.get() != "custom":
        custom_frame.grid_forget()

    main_frame.columnconfigure(1, weight=1)

    # ── 按钮 ──
    btn_frame = ttk.Frame(main_frame)
    btn_frame.grid(row=row, column=0, columnspan=3, pady=(15, 0))

    def on_ok() -> None:
        result["study_root"] = study_root_var.get().strip().replace("\\", "/")
        result["recording_format"] = recording_format_var.get()
        result["csv_marker_mode"] = csv_marker_mode_var.get()
        result["participant"] = entries["participant"].get().strip()
        result["session"] = entries["session"].get().strip()
        try:
            result["run"] = int(entries["run"].get().strip())
        except ValueError:
            result["run"] = 1
        result["trial_mode"] = trial_var.get()
        result["mode"] = mode_var.get()
        result["class_mode"] = class_var.get()
        result["fullscreen"] = fullscreen_var.get()
        # 自定义模式设置
        if mode_var.get() == "custom":
            try:
                result["blocks"] = int(custom_blocks_var.get().strip())
            except ValueError:
                result["blocks"] = 2
            try:
                result["repeats_per_class"] = int(custom_trials_var.get().strip())
            except ValueError:
                result["repeats_per_class"] = 10
        # 从组合选择中获取显示器索引
        sel = monitor_var.get()
        result["display_index"] = monitor_labels.index(sel) if sel in monitor_labels else 0
        # 刷新率
        try:
            result["refresh_rate"] = float(refresh_var.get().strip())
        except ValueError:
            result["refresh_rate"] = float(monitors[result["display_index"]]["refresh_rate"])
        # SSVEP 特定覆盖（仅 trial_mode == mi_ssvep 或 pure_ssvep 时相关）
        result["ssvep_flicker_mode"] = flicker_mode_var.get()
        result["ssvep_waveform"] = waveform_var.get()
        result["ssvep_display_mode"] = ssvep_display_mode_var.get()
        try:
            result["ssvep_left_freq"] = float(left_freq_var.get().strip())
        except ValueError:
            result["ssvep_left_freq"] = 10.0
        try:
            result["ssvep_right_freq"] = float(right_freq_var.get().strip())
        except ValueError:
            result["ssvep_right_freq"] = 15.0
        # P300 特定覆盖（仅 trial_mode == mi_p300 时相关）
        result["p300_flicker_mode"] = p300_flicker_mode_var.get()
        try:
            prob = float(p300_target_prob_var.get().strip())
            result["p300_target_probability"] = max(0.01, min(0.99, prob))
        except ValueError:
            result["p300_target_probability"] = 0.25
        # SSVEP Arousal 特定覆盖（仅 trial_mode == mi_ssvep_arousal 时相关）
        result["ssvep_arousal_freq_mode"] = ssvep_arousal_freq_mode_var.get()
        result["ssvep_arousal_waveform"] = ssvep_arousal_waveform_var.get()
        result["ssvep_arousal_cue_style"] = ssvep_arousal_cue_style_var.get()
        result["ssvep_arousal_task_style"] = ssvep_arousal_task_style_var.get()
        result["ssvep_arousal_arrow_color"] = ssvep_arousal_arrow_color_var.get()
        try:
            result["ssvep_arousal_fixed_freq_hz"] = float(ssvep_arousal_fixed_freq_var.get().strip())
        except ValueError:
            result["ssvep_arousal_fixed_freq_hz"] = 20.0
        try:
            result["ssvep_arousal_freq_min_hz"] = float(ssvep_arousal_freq_min_var.get().strip())
        except ValueError:
            result["ssvep_arousal_freq_min_hz"] = 18.0
        try:
            result["ssvep_arousal_freq_max_hz"] = float(ssvep_arousal_freq_max_var.get().strip())
        except ValueError:
            result["ssvep_arousal_freq_max_hz"] = 25.0
        try:
            result["ssvep_arousal_stimulus_size"] = float(ssvep_arousal_stimulus_size_var.get().strip())
        except ValueError:
            result["ssvep_arousal_stimulus_size"] = 0.35
        try:
            result["ssvep_arousal_dim_opacity"] = float(ssvep_arousal_dim_opacity_var.get().strip())
        except ValueError:
            result["ssvep_arousal_dim_opacity"] = 0.0
        try:
            result["ssvep_arousal_arrow_height"] = float(ssvep_arousal_arrow_height_var.get().strip())
        except ValueError:
            result["ssvep_arousal_arrow_height"] = 0.20
        # SSVEP Serial 特定覆盖（仅 trial_mode == mi_ssvep_serial 时相关）
        result["ssvep_serial_cue_ssvep_mode"] = ssvep_serial_cue_ssvep_mode_var.get()
        result["ssvep_serial_waveform"] = ssvep_serial_waveform_var.get()
        result["ssvep_serial_cue_style"] = ssvep_serial_cue_style_var.get()
        result["ssvep_serial_task_style"] = ssvep_serial_task_style_var.get()
        result["ssvep_serial_display_mode"] = ssvep_serial_display_mode_var.get()
        result["ssvep_serial_arrow_color"] = ssvep_serial_arrow_color_var.get()
        try:
            result["ssvep_serial_cue_ssvep_freq_left_hz"] = float(ssvep_serial_cue_ssvep_freq_left_var.get().strip())
        except ValueError:
            result["ssvep_serial_cue_ssvep_freq_left_hz"] = 10.0
        try:
            result["ssvep_serial_cue_ssvep_freq_right_hz"] = float(ssvep_serial_cue_ssvep_freq_right_var.get().strip())
        except ValueError:
            result["ssvep_serial_cue_ssvep_freq_right_hz"] = 15.0
        try:
            result["ssvep_serial_same_freq_hz"] = float(ssvep_serial_same_freq_var.get().strip())
        except ValueError:
            result["ssvep_serial_same_freq_hz"] = 20.0
        try:
            result["ssvep_serial_cue_ssvep_duration_s"] = float(ssvep_serial_cue_ssvep_duration_var.get().strip())
        except ValueError:
            result["ssvep_serial_cue_ssvep_duration_s"] = 2.0
        try:
            result["ssvep_serial_gap_duration_s"] = float(ssvep_serial_gap_duration_var.get().strip())
        except ValueError:
            result["ssvep_serial_gap_duration_s"] = 2.0
        try:
            result["ssvep_serial_mi_duration_s"] = float(ssvep_serial_mi_duration_var.get().strip())
        except ValueError:
            result["ssvep_serial_mi_duration_s"] = 4.0
        try:
            result["ssvep_serial_stimulus_width"] = float(ssvep_serial_stimulus_width_var.get().strip())
        except ValueError:
            result["ssvep_serial_stimulus_width"] = 0.35
        try:
            result["ssvep_serial_stimulus_height"] = float(ssvep_serial_stimulus_height_var.get().strip())
        except ValueError:
            result["ssvep_serial_stimulus_height"] = 0.35
        try:
            result["ssvep_serial_border_width"] = float(ssvep_serial_border_width_var.get().strip())
        except ValueError:
            result["ssvep_serial_border_width"] = 4.0
        try:
            result["ssvep_serial_dim_opacity"] = float(ssvep_serial_dim_opacity_var.get().strip())
        except ValueError:
            result["ssvep_serial_dim_opacity"] = 0.0
        try:
            result["ssvep_serial_arrow_height"] = float(ssvep_serial_arrow_height_var.get().strip())
        except ValueError:
            result["ssvep_serial_arrow_height"] = 0.20
        # SSVEP RT 特定覆盖（仅 trial_mode == mi_ssvep_rt 时相关）
        result["ssvep_rt_mi_enabled"] = ssvep_rt_mi_enabled_var.get()
        result["ssvep_rt_mi_checkpoint_path"] = ssvep_rt_checkpoint_var.get()
        result["ssvep_rt_flicker_mode"] = ssvep_rt_flicker_mode_var.get()
        result["ssvep_rt_waveform"] = ssvep_rt_waveform_var.get()
        result["ssvep_rt_display_mode"] = ssvep_rt_display_mode_var.get()
        try:
            result["ssvep_rt_classifier_window_s"] = float(ssvep_rt_window_var.get() or "1.5")
        except ValueError:
            result["ssvep_rt_classifier_window_s"] = 1.5
        try:
            result["ssvep_rt_classifier_stride_s"] = float(ssvep_rt_stride_var.get() or "0.25")
        except ValueError:
            result["ssvep_rt_classifier_stride_s"] = 0.25
        try:
            result["ssvep_rt_confidence_threshold"] = float(ssvep_rt_conf_var.get() or "0.15")
        except ValueError:
            result["ssvep_rt_confidence_threshold"] = 0.15
        try:
            result["ssvep_rt_left_freq_hz"] = float(ssvep_rt_left_freq_var.get() or "10.0")
        except ValueError:
            result["ssvep_rt_left_freq_hz"] = 10.0
        try:
            result["ssvep_rt_right_freq_hz"] = float(ssvep_rt_right_freq_var.get() or "15.0")
        except ValueError:
            result["ssvep_rt_right_freq_hz"] = 15.0
        result["ssvep_rt_enable_diag"] = ssvep_rt_enable_diag_var.get()
        root.destroy()

    def on_cancel() -> None:
        cancelled[0] = True
        root.destroy()

    ttk.Button(btn_frame, text="确定 (OK)", command=on_ok, width=12).pack(side="left", padx=5)
    ttk.Button(btn_frame, text="取消 (Cancel)", command=on_cancel, width=12).pack(side="left", padx=5)

    root.protocol("WM_DELETE_WINDOW", on_cancel)

    # 聚焦第一个输入
    entries["participant"].focus_set()
    entries["participant"].select_range(0, "end")

    # 绑定 Enter 到 OK，Escape 到 Cancel
    root.bind("<Return>", lambda _e: on_ok())
    root.bind("<Escape>", lambda _e: on_cancel())

    root.mainloop()

    if cancelled[0] or not result:
        return None

    # 保存以供下次使用
    _save_last_session(**result)

    # 用对话框值覆盖参数
    args.participant = result["participant"]
    args.session = result["session"]
    args.run = result["run"]
    args.trial_mode = result["trial_mode"]
    args.mode = result["mode"]
    args.class_mode = result["class_mode"]
    args.display_index = result["display_index"]
    args.study_root = result["study_root"]
    args.recording_format = result.get("recording_format", "xdf")
    args.csv_marker_mode = result.get("csv_marker_mode", "legacy")
    args.refresh_rate = result["refresh_rate"]
    args.fullscreen = result.get("fullscreen", True)
    if "blocks" in result:
        args.blocks = result["blocks"]
    if "repeats_per_class" in result:
        args.repeats_per_class = result["repeats_per_class"]
    # 来自对话框的 SSVEP 特定覆盖
    if "ssvep_flicker_mode" in result:
        args.ssvep_flicker_mode = result["ssvep_flicker_mode"]
    if "ssvep_waveform" in result:
        args.ssvep_waveform = result["ssvep_waveform"]
    if "ssvep_display_mode" in result:
        args.ssvep_display_mode = result["ssvep_display_mode"]
    if "ssvep_left_freq" in result:
        args.ssvep_left_freq = result["ssvep_left_freq"]
    if "ssvep_right_freq" in result:
        args.ssvep_right_freq = result["ssvep_right_freq"]
    # 来自对话框的 P300 特定覆盖
    if "p300_flicker_mode" in result:
        args.p300_flicker_mode = result["p300_flicker_mode"]
    if "p300_target_probability" in result:
        args.p300_target_probability = result["p300_target_probability"]
    # 来自对话框的 SSVEP Arousal 特定覆盖
    if "ssvep_arousal_freq_mode" in result:
        args.ssvep_arousal_freq_mode = result["ssvep_arousal_freq_mode"]
    if "ssvep_arousal_fixed_freq_hz" in result:
        args.ssvep_arousal_fixed_freq_hz = result["ssvep_arousal_fixed_freq_hz"]
    if "ssvep_arousal_freq_min_hz" in result:
        args.ssvep_arousal_freq_min_hz = result["ssvep_arousal_freq_min_hz"]
    if "ssvep_arousal_freq_max_hz" in result:
        args.ssvep_arousal_freq_max_hz = result["ssvep_arousal_freq_max_hz"]
    if "ssvep_arousal_waveform" in result:
        args.ssvep_arousal_waveform = result["ssvep_arousal_waveform"]
    if "ssvep_arousal_cue_style" in result:
        args.ssvep_arousal_cue_style = result["ssvep_arousal_cue_style"]
    if "ssvep_arousal_task_style" in result:
        args.ssvep_arousal_task_style = result["ssvep_arousal_task_style"]
    if "ssvep_arousal_stimulus_size" in result:
        args.ssvep_arousal_stimulus_size = result["ssvep_arousal_stimulus_size"]
    if "ssvep_arousal_dim_opacity" in result:
        args.ssvep_arousal_dim_opacity = result["ssvep_arousal_dim_opacity"]
    if "ssvep_arousal_arrow_color" in result:
        args.ssvep_arousal_arrow_color = result["ssvep_arousal_arrow_color"]
    if "ssvep_arousal_arrow_height" in result:
        args.ssvep_arousal_arrow_height = result["ssvep_arousal_arrow_height"]
    # 来自对话框的 SSVEP Serial 特定覆盖
    if "ssvep_serial_cue_ssvep_mode" in result:
        args.ssvep_serial_cue_ssvep_mode = result["ssvep_serial_cue_ssvep_mode"]
    if "ssvep_serial_waveform" in result:
        args.ssvep_serial_waveform = result["ssvep_serial_waveform"]
    if "ssvep_serial_cue_style" in result:
        args.ssvep_serial_cue_style = result["ssvep_serial_cue_style"]
    if "ssvep_serial_task_style" in result:
        args.ssvep_serial_task_style = result["ssvep_serial_task_style"]
    if "ssvep_serial_display_mode" in result:
        args.ssvep_serial_display_mode = result["ssvep_serial_display_mode"]
    if "ssvep_serial_arrow_color" in result:
        args.ssvep_serial_arrow_color = result["ssvep_serial_arrow_color"]
    if "ssvep_serial_cue_ssvep_freq_left_hz" in result:
        args.ssvep_serial_cue_ssvep_freq_left_hz = result["ssvep_serial_cue_ssvep_freq_left_hz"]
    if "ssvep_serial_cue_ssvep_freq_right_hz" in result:
        args.ssvep_serial_cue_ssvep_freq_right_hz = result["ssvep_serial_cue_ssvep_freq_right_hz"]
    if "ssvep_serial_same_freq_hz" in result:
        args.ssvep_serial_same_freq_hz = result["ssvep_serial_same_freq_hz"]
    if "ssvep_serial_cue_ssvep_duration_s" in result:
        args.ssvep_serial_cue_ssvep_duration_s = result["ssvep_serial_cue_ssvep_duration_s"]
    if "ssvep_serial_gap_duration_s" in result:
        args.ssvep_serial_gap_duration_s = result["ssvep_serial_gap_duration_s"]
    if "ssvep_serial_mi_duration_s" in result:
        args.ssvep_serial_mi_duration_s = result["ssvep_serial_mi_duration_s"]
    if "ssvep_serial_stimulus_width" in result:
        args.ssvep_serial_stimulus_width = result["ssvep_serial_stimulus_width"]
    if "ssvep_serial_stimulus_height" in result:
        args.ssvep_serial_stimulus_height = result["ssvep_serial_stimulus_height"]
    if "ssvep_serial_border_width" in result:
        args.ssvep_serial_border_width = result["ssvep_serial_border_width"]
    if "ssvep_serial_dim_opacity" in result:
        args.ssvep_serial_dim_opacity = result["ssvep_serial_dim_opacity"]
    if "ssvep_serial_arrow_height" in result:
        args.ssvep_serial_arrow_height = result["ssvep_serial_arrow_height"]
    # 来自对话框的 SSVEP RT 特定覆盖
    if "ssvep_rt_mi_enabled" in result:
        args.ssvep_rt_mi_enabled = result["ssvep_rt_mi_enabled"]
    if "ssvep_rt_mi_checkpoint_path" in result:
        args.ssvep_rt_mi_checkpoint_path = result["ssvep_rt_mi_checkpoint_path"]
    if "ssvep_rt_classifier_window_s" in result:
        args.ssvep_rt_classifier_window_s = result["ssvep_rt_classifier_window_s"]
    if "ssvep_rt_classifier_stride_s" in result:
        args.ssvep_rt_classifier_stride_s = result["ssvep_rt_classifier_stride_s"]
    if "ssvep_rt_confidence_threshold" in result:
        args.ssvep_rt_confidence_threshold = result["ssvep_rt_confidence_threshold"]
    if "ssvep_rt_left_freq_hz" in result:
        args.ssvep_rt_left_freq_hz = result["ssvep_rt_left_freq_hz"]
    if "ssvep_rt_right_freq_hz" in result:
        args.ssvep_rt_right_freq_hz = result["ssvep_rt_right_freq_hz"]
    if "ssvep_rt_flicker_mode" in result:
        args.ssvep_rt_flicker_mode = result["ssvep_rt_flicker_mode"]
    if "ssvep_rt_waveform" in result:
        args.ssvep_rt_waveform = result["ssvep_rt_waveform"]
    if "ssvep_rt_display_mode" in result:
        args.ssvep_rt_display_mode = result["ssvep_rt_display_mode"]
    if "ssvep_rt_enable_diag" in result:
        args.ssvep_rt_enable_diag = result["ssvep_rt_enable_diag"]
    return args


# 入口流程：CLI 解析 → 读取默认值 → 弹出对话框（所有参数的主配置入口）→ build_config → run_session
def main() -> int:
    try:
        args = parse_args()
        defaults = _load_dialog_defaults(args)
        args = show_session_dialog(args, defaults)
        if args is None:
            print("用户取消，退出。")
            return 0
        config = build_config(args)
        return run_session(config)
    except Exception as exc:
        print(f"[FATAL] main() 未捕获异常: {exc!r}")
        import traceback
        traceback.print_exc()
        return 1


# 三种启动模式：全屏阻塞 / 检测-尝试 / 直接开始
if __name__ == "__main__":
    raise SystemExit(main())
