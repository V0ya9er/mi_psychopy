from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Thread-safe ring buffer for real-time EEG data."""

    def __init__(self, capacity_samples: int, n_channels: int) -> None:
        self._buffer = np.zeros((capacity_samples, n_channels), dtype=np.float64)
        self._capacity = capacity_samples
        self._n_channels = n_channels
        self._write_pos = 0
        self._count = 0
        self._lock = threading.Lock()

    def append(self, samples: np.ndarray) -> None:
        """Append new samples. Shape: (n_samples, n_channels)."""
        with self._lock:
            n = samples.shape[0]
            if n >= self._capacity:
                self._buffer[:] = samples[-self._capacity:]
                self._write_pos = 0
                self._count = self._capacity
                return
            end_pos = self._write_pos + n
            if end_pos <= self._capacity:
                self._buffer[self._write_pos:end_pos] = samples
            else:
                first = self._capacity - self._write_pos
                self._buffer[self._write_pos:] = samples[:first]
                self._buffer[:n - first] = samples[first:]
            self._write_pos = end_pos % self._capacity
            self._count = min(self._count + n, self._capacity)

    def get_window(self, n_samples: int) -> np.ndarray | None:
        """Get the most recent n_samples. Returns None if insufficient data."""
        with self._lock:
            if self._count < n_samples:
                return None
            start = (self._write_pos - n_samples) % self._capacity
            if start + n_samples <= self._capacity:
                return self._buffer[start:start + n_samples].copy()
            else:
                first = self._capacity - start
                return np.concatenate([self._buffer[start:], self._buffer[:n_samples - first]])

    def clear(self) -> None:
        with self._lock:
            self._write_pos = 0
            self._count = 0

    @property
    def available(self) -> int:
        with self._lock:
            return self._count

    @property
    def is_full(self) -> bool:
        with self._lock:
            return self._count == self._capacity
