from __future__ import annotations

import os
import platform
import re
import subprocess
import time
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None  # type: ignore

SAFE_PATHS = [Path("C:\\Users"), Path("C:\\Temp"), Path(os.path.abspath("."))]

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKF]")


def _safe(p: str) -> bool:
    path = Path(p).resolve()
    return any(str(path).startswith(str(s)) for s in SAFE_PATHS)


def _run_powershell(cmd: str, timeout: int = 10) -> str:
    try:
        proc = subprocess.run(
            ["powershell", "-NoLogo", "-NoProfile", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return "powershell not available"
    except subprocess.TimeoutExpired:
        return "command timed out"
    out = proc.stdout.strip() or proc.stderr.strip()
    return ANSI_RE.sub("", out)


def cpu_model() -> str:
    """{"name": "cpu_model"}"""

    for cmd in [
        "Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name",
        "wmic cpu get name",
    ]:
        out = _run_powershell(cmd, timeout=4)
        if out and "not" not in out.lower():
            lines = [line for line in out.splitlines() if line.strip()]
            if lines:
                return lines[-1].strip()
    return platform.processor() or "Unknown"


def sys_snapshot() -> Dict[str, Any]:
    """{"name": "sys_snapshot"}"""

    if psutil is None:
        return {
            "cpu_percent": 0.0,
            "ram_percent": 0.0,
            "disk_percent_free_overall": 0.0,
            "uptime_h": 0.0,
        }

    disk = psutil.disk_usage("C:\\" if os.name == "nt" else "/")
    up_hours = (time.time() - psutil.boot_time()) / 3600.0
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "ram_percent": psutil.virtual_memory().percent,
        "disk_percent_free_overall": 100.0 - disk.percent,
        "uptime_h": round(up_hours, 2),
    }


def run_ps(command: str, timeout: int = 10) -> str:
    """{"name": "run_ps"}"""

    if any(bad in command for bad in ["rm ", "del ", "Remove-Item"]):
        return "blocked"
    return _run_powershell(command, timeout)


def browse(path: str) -> List[str]:
    """{"name": "browse"}"""
    p = Path(path)
    if not _safe(str(p)):
        return []
    items = []
    for child in p.iterdir():
        items.append(child.name + ("/" if child.is_dir() else ""))
    return sorted(items)


def list_dir(path: str) -> List[str]:
    """{"name": "list_dir"}"""
    return browse(path)


def read_file(path: str, max_bytes: int = 100_000) -> dict:
    """{"name": "read_file"}"""
    p = Path(path)
    if not _safe(str(p)):
        return {"error": "path unsafe"}
    try:
        with open(p, "rb") as fh:
            data = fh.read(max_bytes)
        return {"content": data.decode(errors="replace")}
    except Exception as exc:  # pragma: no cover - file errors
        return {"error": str(exc)}


def write_file(path: str, content: str) -> dict:
    """{"name": "write_file"}"""
    p = Path(path)
    if not _safe(str(p)):
        return {"error": "access denied"}
    try:
        p.write_text(content, encoding="utf-8")
        return {"status": "success"}
    except Exception as exc:  # pragma: no cover - file errors
        return {"error": str(exc)}


def search_files(pattern: str, path: str) -> List[str]:
    """{"name": "search_files"}"""
    root = Path(path)
    if not _safe(str(root)):
        return []
    return [str(p) for p in root.rglob(pattern) if p.is_file()]


# New helpers following extended schema
def get_cpu_model() -> dict:
    """{"name": "get_cpu_model"}"""
    return {"model": cpu_model()}


def get_memory_info() -> dict:
    """{"name": "get_memory_info"}"""
    if psutil is None:
        return {"total_gb": 0.0, "available_gb": 0.0}
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024**3), 2),
        "available_gb": round(vm.available / (1024**3), 2),
    }


def ram_info() -> dict:
    """{"name": "ram_info"}"""
    if psutil is None:
        return {"total_gb": 0.0, "free_gb": 0.0, "percent": 0.0}
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024**3), 2),
        "free_gb": round(vm.available / (1024**3), 2),
        "percent": vm.percent,
    }


def list_disks() -> dict:
    """{"name": "list_disks"}"""
    if psutil is None:
        return {"disks": []}
    out: List[dict] = []
    for part in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except Exception:
            continue
        out.append(
            {
                "mount": part.mountpoint,
                "total_gb": round(usage.total / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
        )
    return {"disks": out}


def disk_info(all_drives: bool = True) -> List[dict]:
    """{"name": "disk_info"}"""
    if psutil is None:
        return []
    out = []
    for part in psutil.disk_partitions():
        if not all_drives and part.mountpoint not in ("/", "C:\\"):
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except Exception:
            continue
        out.append(
            {
                "letter": part.device.rstrip("\\"),
                "total_gb": round(usage.total / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
            }
        )
    return out


def get_motherboard_info() -> dict:
    """{"name": "get_motherboard_info"}"""
    if os.name == "nt":
        cmd = "Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product,SerialNumber"
        text = _run_powershell(cmd)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= 3:
            return {"manufacturer": lines[0], "product": lines[1], "serial": lines[2]}
        return {"raw": text}
    base = Path("/sys/class/dmi/id")
    try:
        return {
            "manufacturer": (base / "board_vendor").read_text().strip(),
            "product": (base / "board_name").read_text().strip(),
            "serial": (base / "board_serial").read_text().strip(),
        }
    except Exception:
        return {}


def mb_info() -> dict:
    """{"name": "mb_info"}"""
    info = get_motherboard_info()
    bios = ""
    if os.name == "nt":
        bios = _run_powershell("(Get-CimInstance Win32_BIOS).SMBIOSBIOSVersion")
    else:
        try:
            bios = Path("/sys/class/dmi/id/bios_version").read_text().strip()
        except Exception:
            bios = ""
    info["bios"] = bios
    return info


def gpu_info() -> dict:
    """{"name": "gpu_info"}"""
    if os.name == "nt":
        text = _run_powershell(
            "Get-CimInstance Win32_VideoController | Select-Object -First 1 Name,AdapterRAM"
        )
        parts = [p.strip() for p in text.splitlines() if p.strip()]
        if len(parts) >= 2:
            try:
                vram = round(int(parts[1]) / (1024**3), 2)
            except Exception:
                vram = 0.0
            return {"name": parts[0], "vram_gb": vram}
        return {"raw": text}
    try:
        with open("/proc/driver/nvidia/gpus/0000:00:00.0/information", "r") as fh:
            lines = fh.read().splitlines()
        name = next(
            (
                line.split(":", 1)[1].strip()
                for line in lines
                if line.startswith("Model")
            ),
            "",
        )
        mem = next(
            (
                line.split(":", 1)[1].strip()
                for line in lines
                if line.startswith("Memory")
            ),
            "0",
        )
        vram = float(mem.split()[0])
        return {"name": name, "vram_gb": vram}
    except Exception:
        return {}


def snapshot_system() -> dict:
    """{"name": "snapshot_system"}"""
    snap = sys_snapshot()
    return {
        "cpu_percent": snap.get("cpu_percent", 0.0),
        "ram_percent": snap.get("ram_percent", 0.0),
        "disk_percent_free_overall": snap.get("disk_percent_free_overall", 0.0),
        "uptime_h": snap.get("uptime_h", 0.0),
    }


def list_files(path: str) -> dict:
    """{"name": "list_files"}"""
    items = browse(path)
    if not items:
        return {"error": "path unsafe"}
    return {"files": items}


def run_process(command: str, timeout_seconds: int = 30) -> dict:
    """{"name": "run_process"}"""
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    return {"stdout": proc.stdout, "stderr": proc.stderr, "exit_code": proc.returncode}


TOOL_FUNCS = {
    "cpu_model": cpu_model,
    "ram_info": ram_info,
    "disk_info": disk_info,
    "mb_info": mb_info,
    "gpu_info": gpu_info,
    "sys_snapshot": sys_snapshot,
    "list_dir": list_dir,
    "read_file": read_file,
    "write_file": write_file,
    "search_files": search_files,
    "run_ps": run_ps,
    "run_process": run_process,
    "get_cpu_model": get_cpu_model,
    "get_memory_info": get_memory_info,
    "list_disks": list_disks,
    "get_motherboard_info": get_motherboard_info,
    "snapshot_system": snapshot_system,
    "browse": browse,
    "list_files": list_files,
}


def build_tool_schema(fn: Any) -> Dict[str, Any]:
    """Return OpenAI-compatible schema for ``fn``."""

    raw = json.loads(fn.__doc__ or "{}")
    return {"type": "function", "function": raw}


TOOL_SCHEMAS: List[Dict[str, Any]] = [build_tool_schema(f) for f in TOOL_FUNCS.values()]
