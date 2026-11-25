"""Whether the host ends a command's processes with the command (Windows job objects).

A host may run each command in a job that kills its processes when the job closes and lets none leave (Antigravity,
flags 0x2000, measured 2026-11-21 and 2026-11-25): a detached child the command starts dies when the command returns.
Windows shows a process only the innermost job it is in, so a job that does not kill on close may sit inside one that
does. The one thing that can be read for certain is that a child is in no job at all: only then is it known to outlive
the command. A launcher's own job (`py.exe`, a venv or console-script launcher: children leave it silently, flags
0x3000) is left by the children, so a child outside it answers for the next job out.
"""
import os
import subprocess
import sys

SILENT_BREAKAWAY = 0x1000
HOPS = 3
_NO_JOB, _UNREADABLE = object(), object()


def _own_limits():
    """The limit flags of the innermost job this process is in, `_NO_JOB` outside any job, `_UNREADABLE` when the job
    does not answer."""
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):   # JOBOBJECT_BASIC_LIMIT_INFORMATION
        _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong), ('flags', wintypes.DWORD),
                    ('min_working_set', ctypes.c_size_t), ('max_working_set', ctypes.c_size_t),
                    ('active_processes', wintypes.DWORD), ('affinity', ctypes.c_size_t),
                    ('priority', wintypes.DWORD), ('scheduling', wintypes.DWORD)]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                                 ctypes.c_void_p]
    inside = wintypes.BOOL()
    if not kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(inside)) or not inside.value:
        return _NO_JOB
    limits = BasicLimits()
    if not kernel.QueryInformationJobObject(None, 2, ctypes.byref(limits), ctypes.sizeof(limits), None):
        return _UNREADABLE
    return limits.flags


def ends_children(hops=HOPS):
    """True unless a process this command starts is in no job, so that nothing of the host's can end it with the
    command. A job that keeps its children counts as true whatever its own limits say, and so does one that cannot be
    read: the answer that follows (do the step in the foreground) holds either way."""
    if os.name != 'nt':
        return False
    flags = _own_limits()
    if flags is _NO_JOB:
        return False
    if flags is _UNREADABLE or not flags & SILENT_BREAKAWAY:
        return True
    if hops <= 0:
        return True
    # Started without a launcher of its own, or it would sit in a new launcher job and read that one.
    interpreter = sys.executable if getattr(sys, 'frozen', False) else getattr(sys, '_base_executable', None) or sys.executable
    try:
        answer = subprocess.run([interpreter, str(os.path.abspath(__file__)), str(hops - 1)], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return True
    return answer.stdout.strip() != 'False'


if __name__ == '__main__':
    print(ends_children(int(sys.argv[1])))
