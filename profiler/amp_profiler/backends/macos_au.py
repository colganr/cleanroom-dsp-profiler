"""
Headless macOS AudioUnit (`AudioToolbox` + `CFRunLoop`) Black-Box Profiling Backend.

Implements all critical real-world host guardrails:
  - Explicit `CFRunLoopRunInMode` pumping + multi-block silent flush so JUCE/async plugins
    finish loading internal models and settling `SmoothedValue` ramps.
  - Safe fast process exit hook so AudioUnits with background UI threads do not hang on exit.
  - Strictly black-box audio I/O + standard AudioUnit parameter calls only.
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import os
import platform
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    byref,
    c_byte,
    c_double,
    c_float,
    c_int32,
    c_uint32,
    c_uint64,
    c_void_p,
)
import numpy as np
from .base import ParamInfo, TargetBackend


def fourcc(code: str) -> int:
    raw = code.encode("ascii")
    if len(raw) != 4:
        raise ValueError(f"FourCC must be exactly 4 ASCII characters, got: {code!r}")
    return (raw[0] << 24) | (raw[1] << 16) | (raw[2] << 8) | raw[3]


class _AudioComponentDescription(Structure):
    _fields_ = [
        ("componentType", c_uint32),
        ("componentSubType", c_uint32),
        ("componentManufacturer", c_uint32),
        ("componentFlags", c_uint32),
        ("componentFlagsMask", c_uint32),
    ]


class _AudioStreamBasicDescription(Structure):
    _fields_ = [
        ("mSampleRate", c_double),
        ("mFormatID", c_uint32),
        ("mFormatFlags", c_uint32),
        ("mBytesPerPacket", c_uint32),
        ("mFramesPerPacket", c_uint32),
        ("mBytesPerFrame", c_uint32),
        ("mChannelsPerFrame", c_uint32),
        ("mBitsPerChannel", c_uint32),
        ("mReserved", c_uint32),
    ]


class _AudioBuffer(Structure):
    _fields_ = [
        ("mNumberChannels", c_uint32),
        ("mDataByteSize", c_uint32),
        ("mData", c_void_p),
    ]


class _AudioBufferList2(Structure):
    _fields_ = [
        ("mNumberBuffers", c_uint32),
        ("mBuffers", _AudioBuffer * 2),
    ]


class _AudioTimeStamp(Structure):
    _fields_ = [
        ("mSampleTime", c_double),
        ("mHostTime", c_uint64),
        ("mRateScalar", c_double),
        ("mWordClockTime", c_uint64),
        ("mSMPTETime", c_byte * 24),
        ("mFlags", c_uint32),
        ("mReserved", c_uint32),
    ]


class _AudioUnitParameterInfo(Structure):
    _fields_ = [
        ("name", c_byte * 52),
        ("unitName", c_void_p),
        ("clumpID", c_uint32),
        ("cfNameString", c_void_p),
        ("unit", c_uint32),
        ("minValue", c_float),
        ("maxValue", c_float),
        ("defaultValue", c_float),
        ("flags", c_uint32),
    ]


class _AURenderCallbackStruct(Structure):
    _fields_ = [
        ("inputProc", c_void_p),
        ("inputProcRefCon", c_void_p),
    ]


_RENDER_CB = CFUNCTYPE(
    c_int32,
    c_void_p,
    POINTER(c_uint32),
    POINTER(_AudioTimeStamp),
    c_uint32,
    c_uint32,
    POINTER(_AudioBufferList2),
)


class MacOSAudioUnitBackend(TargetBackend):
    """Hosts any locally installed macOS AudioUnit headlessly at 48 kHz for black-box profiling."""

    def __init__(
        self,
        comp_type: str,
        comp_subtype: str,
        comp_mfg: str,
        sr: int = 48_000,
        block_size: int = 512,
        safe_fast_exit: bool = True,
    ):
        if platform.system() != "Darwin":
            raise RuntimeError(
                "MacOSAudioUnitBackend requires macOS (Darwin). "
                "On Windows/Linux, use WavPairBackend (`profile-wav-pair`) or a cross-platform host."
            )

        self.sr = int(sr)
        self.block_size = int(block_size)
        self.sample_time = 0.0

        self._at = ctypes.cdll.LoadLibrary(ctypes.util.find_library("AudioToolbox"))
        self._cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        self._cf.CFRunLoopRunInMode.argtypes = [c_void_p, c_double, ctypes.c_bool]
        self._kCFRunLoopDefaultMode = c_void_p.in_dll(self._cf, "kCFRunLoopDefaultMode")

        at = self._at
        at.AudioComponentFindNext.argtypes = [c_void_p, POINTER(_AudioComponentDescription)]
        at.AudioComponentFindNext.restype = c_void_p
        at.AudioComponentInstanceNew.argtypes = [c_void_p, POINTER(c_void_p)]
        at.AudioUnitSetProperty.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32, c_void_p, c_uint32]
        at.AudioUnitGetPropertyInfo.argtypes = [
            c_void_p, c_uint32, c_uint32, c_uint32, POINTER(c_uint32), POINTER(c_byte)
        ]
        at.AudioUnitGetProperty.argtypes = [
            c_void_p, c_uint32, c_uint32, c_uint32, c_void_p, POINTER(c_uint32)
        ]
        at.AudioUnitInitialize.argtypes = [c_void_p]
        at.AudioUnitSetParameter.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32, c_float, c_uint32]
        at.AudioUnitGetParameter.argtypes = [c_void_p, c_uint32, c_uint32, c_uint32, POINTER(c_float)]
        at.AudioUnitReset.argtypes = [c_void_p, c_uint32, c_uint32]
        at.AudioUnitRender.argtypes = [
            c_void_p, POINTER(c_uint32), POINTER(_AudioTimeStamp), c_uint32, c_uint32, POINTER(_AudioBufferList2)
        ]

        desc = _AudioComponentDescription(
            fourcc(comp_type), fourcc(comp_subtype), fourcc(comp_mfg), 0, 0
        )
        comp = at.AudioComponentFindNext(None, byref(desc))
        if not comp:
            raise RuntimeError(f"AudioUnit not found: {comp_type}/{comp_subtype}/{comp_mfg}")

        self.inst = c_void_p()
        err = at.AudioComponentInstanceNew(comp, byref(self.inst))
        if err != 0:
            raise RuntimeError(f"AudioComponentInstanceNew failed (OSStatus {err})")

        if safe_fast_exit:
            def _fast_exit():
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:
                    pass
                os._exit(0)
            atexit.register(_fast_exit)

        asbd = _AudioStreamBasicDescription(
            float(self.sr),
            fourcc("lpcm"),
            0x29,  # kAudioFormatFlagsNativeFloatPacked | kAudioFormatFlagIsNonInterleaved
            4, 1, 4, 2, 32, 0,
        )
        for scope in (1, 2):
            at.AudioUnitSetProperty(self.inst, 8, scope, 0, byref(asbd), ctypes.sizeof(asbd))

        max_frames = c_uint32(self.block_size)
        at.AudioUnitSetProperty(self.inst, 14, 0, 0, byref(max_frames), 4)

        self._in_l = (c_float * self.block_size)()
        self._in_r = (c_float * self.block_size)()
        self._out_l = (c_float * self.block_size)()
        self._out_r = (c_float * self.block_size)()

        def _input_cb(_refcon, _flags, _ts, _bus, nframes, io_data):
            abl = io_data.contents
            ctypes.memmove(abl.mBuffers[0].mData, self._in_l, nframes * 4)
            if abl.mNumberBuffers > 1:
                ctypes.memmove(abl.mBuffers[1].mData, self._in_r, nframes * 4)
            return 0

        self._cb_func = _RENDER_CB(_input_cb)
        cb_struct = _AURenderCallbackStruct(ctypes.cast(self._cb_func, c_void_p), None)
        at.AudioUnitSetProperty(self.inst, 23, 1, 0, byref(cb_struct), ctypes.sizeof(cb_struct))

        err = at.AudioUnitInitialize(self.inst)
        if err != 0:
            raise RuntimeError(f"AudioUnitInitialize failed (OSStatus {err})")

    def list_parameters(self) -> list[ParamInfo]:
        size = c_uint32(0)
        writable = c_byte(0)
        err = self._at.AudioUnitGetPropertyInfo(self.inst, 3, 0, 0, byref(size), byref(writable))
        if err != 0 or size.value == 0:
            return []
        count = size.value // 4
        pids = (c_uint32 * count)()
        self._at.AudioUnitGetProperty(self.inst, 3, 0, 0, byref(pids), byref(size))

        params: list[ParamInfo] = []
        for pid in pids:
            info = _AudioUnitParameterInfo()
            inf_sz = c_uint32(ctypes.sizeof(info))
            if self._at.AudioUnitGetProperty(self.inst, 4, 0, pid, byref(info), byref(inf_sz)) == 0:
                name = bytes(info.name).split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                params.append(
                    ParamInfo(
                        pid=int(pid),
                        name=name,
                        min_value=float(info.minValue),
                        max_value=float(info.maxValue),
                        default_value=float(info.defaultValue),
                    )
                )
        return params

    def set_param(self, pid: int, value: float) -> None:
        self._at.AudioUnitSetParameter(self.inst, c_uint32(pid), 0, 0, c_float(value), 0)

    def get_param(self, pid: int) -> float:
        val = c_float(0.0)
        self._at.AudioUnitGetParameter(self.inst, c_uint32(pid), 0, 0, byref(val))
        return float(val.value)

    def _render_block(self) -> None:
        flags = c_uint32(0)
        ts = _AudioTimeStamp()
        ts.mSampleTime = self.sample_time
        ts.mFlags = 1
        abl = _AudioBufferList2()
        abl.mNumberBuffers = 2
        abl.mBuffers[0].mNumberChannels = 1
        abl.mBuffers[0].mDataByteSize = self.block_size * 4
        abl.mBuffers[0].mData = ctypes.cast(self._out_l, c_void_p)
        abl.mBuffers[1].mNumberChannels = 1
        abl.mBuffers[1].mDataByteSize = self.block_size * 4
        abl.mBuffers[1].mData = ctypes.cast(self._out_r, c_void_p)
        self._at.AudioUnitRender(self.inst, byref(flags), byref(ts), 0, self.block_size, byref(abl))
        self.sample_time += self.block_size

    def settle(self, full: bool = False) -> None:
        """
        Pumps CFRunLoop and renders silent blocks so async model loaders and parameter
        smoothers finish before capture. Use `full=True` on worker startup or mode switches.
        """
        ctypes.memset(self._in_l, 0, self.block_size * 4)
        ctypes.memset(self._in_r, 0, self.block_size * 4)
        passes = 10 if full else 2
        blocks = 28 if full else 6
        for _ in range(passes):
            self._cf.CFRunLoopRunInMode(self._kCFRunLoopDefaultMode, 0.048, False)
            for _ in range(blocks):
                self._render_block()
        self._at.AudioUnitReset(self.inst, 0, 0)
        for _ in range(6):
            self._render_block()

    def render_mono(self, x_in: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x_in, dtype=np.float32)
        n = len(x)
        out = np.zeros(n, dtype=np.float32)
        bs = self.block_size
        pos = 0
        while pos < n:
            take = min(bs, n - pos)
            ctypes.memmove(self._in_l, x[pos : pos + take].ctypes.data, take * 4)
            ctypes.memmove(self._in_r, x[pos : pos + take].ctypes.data, take * 4)
            if take < bs:
                for i in range(take, bs):
                    self._in_l[i] = 0.0
                    self._in_r[i] = 0.0
            self._render_block()
            out[pos : pos + take] = np.frombuffer(self._out_l, dtype=np.float32, count=take)
            pos += take
        return out
