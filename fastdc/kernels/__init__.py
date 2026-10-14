"""Triton kernels of FastDC: paged cache writes, attention on the compressed cache and its task schedules.

The semantics are those of fastdc.kernels.reference on the layout of fastdc.kernels.layout.
"""

from fastdc.kernels.attention import paged_attention
from fastdc.kernels.cache import write_kv
from fastdc.kernels.schedule import Schedule, ScheduleWorkspace, build_schedules

__all__ = ["Schedule", "ScheduleWorkspace", "build_schedules", "paged_attention", "write_kv"]
