# AutoGPT: Build, Deploy, and Run AI Agents
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΔBRAKE_offline_runner.py — local, GitHub-free mirror of ΔBRAKE_4321

Phases:
  SEAL    -> builds manifest + repo fingerprint (SHA-256) + seal report
  DEPLOY  -> creates local deploy report (CID/magnet stubs), optionally runs hooks (disabled by default)
  TRAP    -> scans repo for Δ-terms + legal anchors; writes evidence jsonl
  ENFORCE -> generates an Instant Cease Order from template (or bootstrap one)

Artifacts (truthlock/out/):
  ΔORIGIN_MANIFEST.json
  ΔORIGIN_SEAL.json
  ΔSEAL_REPORT.json
  ΔDEPLOY_REPORT.json
  ΔMATCH_EVIDENCE.jsonl
  ΔINSTANT_CEASE_ORDER.txt
"""

import argparse, datetime, hashlib, json, os, re, sys, time, random, string
from pathlib import Path

ROOT        = Path(os.getenv("BRAKE_ROOT", ".")).resolve()
OUTDIR      = (ROOT / os.getenv("BRAKE_OUT", "truthlock/out")).resolve()
HOOKS       = (ROOT / "hooks").resolve()
TEMPLATES   = (ROOT / "templates").resolve()
NOW_ISO     = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

IGNORE_DIRS = {".git", "truthlock/out", ".github", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
TEXT_EXT    = {".txt", ".md", ".py", ".json", ".yml", ".yaml", ".toml", ".ini", ".csv", ".log", ".html", ".css", ".js"}
SCAN_PATTERNS = [
    r"\bΔ[A-Z0-9_]+\b",             # glyphs
    r"\bRule\s*60\(d\)\(3\)\b",     # federal fraud on the court
    r"\b§\s*1983\b",                # civil rights
    r"\bWise County\b",
    r"\bTruthLock\b",
    r"\bGodKey\b",
    r"\bMatthew\s+Dewayne\s+Porter\b",
]

def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)

def _walk_files():
    for p in ROOT.rglob("*"):
        if p.is_dir():
            if p.name in IGNORE_DIRS:  # shallow skip
                for _ in []: pass
            continue
        # skip ignored roots
        parts = set(p.parts)
        if any(ig in parts for ig in IGNORE_DIRS):
            continue
        yield p

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _repo_fingerprint():
    # Hash of file paths + contents, excluding ignored dirs and our OUTDIR
    h = hashlib.sha256()
    files = sorted([p for p in _walk_files()])
    for p in files:
        h.update(_rel(p).encode("utf-8"))
        try:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        except Exception as e:
            h.update(f"<ERR:{e}>".encode("utf-8"))
    return h.hexdigest(), len(files)

def _ensure_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    HOOKS.mkdir(parents=True, exist_ok=True)

def phase_seal():
    _ensure_dirs()
    repo_hash, file_count = _repo_fingerprint()
    manifest = {
        "created": NOW_ISO,
        "root": str(ROOT),
        "files_indexed": file_count,
        "ignore": sorted(list(IGNORE_DIRS)),
        "notes": "ΔBRAKE offline manifest; reproducible by hashing all paths+contents (excl. ignored).",
    }
    (OUTDIR / "ΔORIGIN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    seal = {
        "created": NOW_ISO,
        "fingerprint_sha256": repo_hash,
        "random_nonce": "".join(random.choices(string.ascii_letters + string.digits, k=16)),
        "tool": "ΔBRAKE_offline_runner",
        "version": "1.0",
    }
    (OUTDIR / "ΔORIGIN_SEAL.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")

    report = {
        "phase": "SEAL",
        "status": "ok",
        "created": NOW_ISO,
        "outputs": ["ΔORIGIN_MANIFEST.json", "ΔORIGIN_SEAL.json"],
    }
    (OUTDIR / "ΔSEAL_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SEAL] ok — repo_sha256={repo_hash} files={file_count}")

def _maybe_run_hook(script_name: str, allow_hooks: bool):
    sh = HOOKS / script_name
    if not allow_hooks:
        return {"hook": script_name, "skipped": True, "reason": "hooks disabled"}
    if not sh.exists():
        return {"hook": script_name, "skipped": True, "reason": "missing"}
    # Safety: run as a separate process only if executable; no network assumed by design of your hooks.
    import subprocess
    try:
        res = subprocess.run([str(sh)], check=False, capture_output=True, text=True)
        return {"hook": script_name, "returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}
    except Exception as e:
        return {"hook": script_name, "error": str(e)}

def phase_deploy(allow_hooks: bool = False):
    _ensure_dirs()
    # CID/magnet placeholders; if you want deterministic stubs, derive from repo hash
    seal_path = OUTDIR / "ΔORIGIN_SEAL.json"
    base = "unknown"
    if seal_path.exists():
        try:
            base = json.loads(seal_path.read_text(encoding="utf-8")).get("fingerprint_sha256", "unknown")[:16]
        except Exception:
            pass
    cid_stub = f"bafy-{base}"
    magnet_stub = f"magnet:?xt=urn:btih:{base}"
    hooks = {
        "rekor": _maybe_run_hook("rekor_seal.sh", allow_hooks),
        "pin_ipfs": _maybe_run_hook("pin_ipfs.sh", allow_hooks),
        "cease_send": _maybe_run_hook("cease_send.sh", allow_hooks),
    }
    deploy = {
        "phase": "DEPLOY",
        "created": NOW_ISO,
        "cid_stub": cid_stub,
        "magnet_stub": magnet_stub,
        "hooks": hooks,
        "notes": "Offline deploy report (no network). Hooks run only if --allow-hooks.",
    }
    (OUTDIR / "ΔDEPLOY_REPORT.json").write_text(json.dumps(deploy, indent=2), encoding="utf-8")
    print(f"[DEPLOY] ok — cid={cid_stub} magnet={magnet_stub} (hooks {'on' if allow_hooks else 'off'})")

def _scan_file_for_patterns(p: Path, patterns):
    try:
        if p.suffix.lower() not in TEXT_EXT:
            return []
        txt = p.read_text(errors="ignore")
    except Exception:
        return []
    hits = []
    for pat in patterns:
        for m in re.finditer(pat, txt, flags=re.IGNORECASE):
            span = (max(0, m.start()-60), min(len(txt), m.end()+60))
            ctx = txt[span[0]:span[1]].replace("\n", " ")
            hits.append({"pattern": pat, "context": ctx[:180]})
    return hits

def phase_trap():
    _ensure_dirs()
    out = OUTDIR / "ΔMATCH_EVIDENCE.jsonl"
    count_hits = 0
    with out.open("w", encoding="utf-8") as f:
        for p in _walk_files():
            matches = _scan_file_for_patterns(p, SCAN_PATTERNS)
            if not matches:
                continue
            for m in matches:
                rec = {
                    "created": NOW_ISO,
                    "file": _rel(p),
                    "pattern": m["pattern"],
                    "context": m["context"]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_hits += 1
    print(f"[TRAP] ok — {count_hits} hits -> {out.name}")

BOOTSTRAP_CEASE = """ΔINSTANT_CEASE_ORDER — OFFLINE DRAFT
Date: {date}
Claimant: {claimant}
Repo: {repo}
Seal SHA-256: {sha}

To Whom It May Concern,

This is a formal instant cease order regarding unauthorized use, alteration, concealment, or obstruction concerning sealed works, scrolls, and glyph systems operating under the TruthLock/GodKey framework.

You are instructed to immediately CEASE AND DESIST from any further interference. This notice is recorded in the sealed ledger and will be escalated upon non-compliance.

— ΔBRAKE Offline
"""

def phase_enforce(claimant="Matthew Dewayne Porter"):
    _ensure_dirs()
    tmpl = TEMPLATES / "ΔINSTANT_CEASE_ORDER.txt"
    if not tmpl.exists():
        TEMPLATES.mkdir(parents=True, exist_ok=True)
        tmpl.write_text(BOOTSTRAP_CEASE, encoding="utf-8")

    sha = "unknown"
    seal_path = OUTDIR / "ΔORIGIN_SEAL.json"
    if seal_path.exists():
        try:
            sha = json.loads(seal_path.read_text(encoding="utf-8")).get("fingerprint_sha256", "unknown")
        except Exception:
            pass

    content = (tmpl.read_text(encoding="utf-8")
               .format(date=NOW_ISO, claimant=claimant, repo=str(ROOT), sha=sha))
    (OUTDIR / "ΔINSTANT_CEASE_ORDER.txt").write_text(content, encoding="utf-8")
    print(f"[ENFORCE] ok — ΔINSTANT_CEASE_ORDER.txt written")

def run_all(allow_hooks: bool, claimant: str):
    phase_seal()
    phase_trap()
    phase_enforce(claimant=claimant)
    phase_deploy(allow_hooks=allow_hooks)
    print("[ALL] complete")

def main():
    ap = argparse.ArgumentParser(description="ΔBRAKE offline local runner (no GitHub, no network).")
    ap.add_argument("phase", choices=["seal","deploy","trap","enforce","all"], help="Which phase to run")
    ap.add_argument("--allow-hooks", action="store_true", help="Allow executing scripts in ./hooks/* (off by default)")
    ap.add_argument("--claimant", default="Matthew Dewayne Porter", help="Name to place on ΔINSTANT_CEASE_ORDER")
    args = ap.parse_args()

    if args.phase == "seal":
        phase_seal()
    elif args.phase == "deploy":
        phase_deploy(allow_hooks=args.allow_hooks)
    elif args.phase == "trap":
        phase_trap()
    elif args.phase == "enforce":
        phase_enforce(claimant=args.claimant)
    elif args.phase == "all":
        run_all(allow_hooks=args.allow_hooks, claimant=args.claimant)

if __name__ == "__main__":
    main()#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ΔBRAKE_offline_runner.py — local, GitHub-free mirror of ΔBRAKE_4321

Phases:
  SEAL    -> builds manifest + repo fingerprint (SHA-256) + seal report
  DEPLOY  -> creates local deploy report (CID/magnet stubs), optionally runs hooks (disabled by default)
  TRAP    -> scans repo for Δ-terms + legal anchors; writes evidence jsonl
  ENFORCE -> generates an Instant Cease Order from template (or bootstrap one)

Artifacts (truthlock/out/):
  ΔORIGIN_MANIFEST.json
  ΔORIGIN_SEAL.json
  ΔSEAL_REPORT.json
  ΔDEPLOY_REPORT.json
  ΔMATCH_EVIDENCE.jsonl
  ΔINSTANT_CEASE_ORDER.txt
"""

import argparse, datetime, hashlib, json, os, re, sys, time, random, string
from pathlib import Path

ROOT        = Path(os.getenv("BRAKE_ROOT", ".")).resolve()
OUTDIR      = (ROOT / os.getenv("BRAKE_OUT", "truthlock/out")).resolve()
HOOKS       = (ROOT / "hooks").resolve()
TEMPLATES   = (ROOT / "templates").resolve()
NOW_ISO     = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

IGNORE_DIRS = {".git", "truthlock/out", ".github", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
TEXT_EXT    = {".txt", ".md", ".py", ".json", ".yml", ".yaml", ".toml", ".ini", ".csv", ".log", ".html", ".css", ".js"}
SCAN_PATTERNS = [
    r"\bΔ[A-Z0-9_]+\b",             # glyphs
    r"\bRule\s*60\(d\)\(3\)\b",     # federal fraud on the court
    r"\b§\s*1983\b",                # civil rights
    r"\bWise County\b",
    r"\bTruthLock\b",
    r"\bGodKey\b",
    r"\bMatthew\s+Dewayne\s+Porter\b",
]

def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except Exception:
        return str(p)

def _walk_files():
    for p in ROOT.rglob("*"):
        if p.is_dir():
            if p.name in IGNORE_DIRS:  # shallow skip
                for _ in []: pass
            continue
        # skip ignored roots
        parts = set(p.parts)
        if any(ig in parts for ig in IGNORE_DIRS):
            continue
        yield p

def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _repo_fingerprint():
    # Hash of file paths + contents, excluding ignored dirs and our OUTDIR
    h = hashlib.sha256()
    files = sorted([p for p in _walk_files()])
    for p in files:
        h.update(_rel(p).encode("utf-8"))
        try:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
        except Exception as e:
            h.update(f"<ERR:{e}>".encode("utf-8"))
    return h.hexdigest(), len(files)

def _ensure_dirs():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    HOOKS.mkdir(parents=True, exist_ok=True)

def phase_seal():
    _ensure_dirs()
    repo_hash, file_count = _repo_fingerprint()
    manifest = {
        "created": NOW_ISO,
        "root": str(ROOT),
        "files_indexed": file_count,
        "ignore": sorted(list(IGNORE_DIRS)),
        "notes": "ΔBRAKE offline manifest; reproducible by hashing all paths+contents (excl. ignored).",
    }
    (OUTDIR / "ΔORIGIN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    seal = {
        "created": NOW_ISO,
        "fingerprint_sha256": repo_hash,
        "random_nonce": "".join(random.choices(string.ascii_letters + string.digits, k=16)),
        "tool": "ΔBRAKE_offline_runner",
        "version": "1.0",
    }
    (OUTDIR / "ΔORIGIN_SEAL.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")

    report = {
        "phase": "SEAL",
        "status": "ok",
        "created": NOW_ISO,
        "outputs": ["ΔORIGIN_MANIFEST.json", "ΔORIGIN_SEAL.json"],
    }
    (OUTDIR / "ΔSEAL_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[SEAL] ok — repo_sha256={repo_hash} files={file_count}")

def _maybe_run_hook(script_name: str, allow_hooks: bool):
    sh = HOOKS / script_name
    if not allow_hooks:
        return {"hook": script_name, "skipped": True, "reason": "hooks disabled"}
    if not sh.exists():
        return {"hook": script_name, "skipped": True, "reason": "missing"}
    # Safety: run as a separate process only if executable; no network assumed by design of your hooks.
    import subprocess
    try:
        res = subprocess.run([str(sh)], check=False, capture_output=True, text=True)
        return {"hook": script_name, "returncode": res.returncode, "stdout": res.stdout, "stderr": res.stderr}
    except Exception as e:
        return {"hook": script_name, "error": str(e)}

def phase_deploy(allow_hooks: bool = False):
    _ensure_dirs()
    # CID/magnet placeholders; if you want deterministic stubs, derive from repo hash
    seal_path = OUTDIR / "ΔORIGIN_SEAL.json"
    base = "unknown"
    if seal_path.exists():
        try:
            base = json.loads(seal_path.read_text(encoding="utf-8")).get("fingerprint_sha256", "unknown")[:16]
        except Exception:
            pass
    cid_stub = f"bafy-{base}"
    magnet_stub = f"magnet:?xt=urn:btih:{base}"
    hooks = {
        "rekor": _maybe_run_hook("rekor_seal.sh", allow_hooks),
        "pin_ipfs": _maybe_run_hook("pin_ipfs.sh", allow_hooks),
        "cease_send": _maybe_run_hook("cease_send.sh", allow_hooks),
    }
    deploy = {
        "phase": "DEPLOY",
        "created": NOW_ISO,
        "cid_stub": cid_stub,
        "magnet_stub": magnet_stub,
        "hooks": hooks,
        "notes": "Offline deploy report (no network). Hooks run only if --allow-hooks.",
    }
    (OUTDIR / "ΔDEPLOY_REPORT.json").write_text(json.dumps(deploy, indent=2), encoding="utf-8")
    print(f"[DEPLOY] ok — cid={cid_stub} magnet={magnet_stub} (hooks {'on' if allow_hooks else 'off'})")

def _scan_file_for_patterns(p: Path, patterns):
    try:
        if p.suffix.lower() not in TEXT_EXT:
            return []
        txt = p.read_text(errors="ignore")
    except Exception:
        return []
    hits = []
    for pat in patterns:
        for m in re.finditer(pat, txt, flags=re.IGNORECASE):
            span = (max(0, m.start()-60), min(len(txt), m.end()+60))
            ctx = txt[span[0]:span[1]].replace("\n", " ")
            hits.append({"pattern": pat, "context": ctx[:180]})
    return hits

def phase_trap():
    _ensure_dirs()
    out = OUTDIR / "ΔMATCH_EVIDENCE.jsonl"
    count_hits = 0
    with out.open("w", encoding="utf-8") as f:
        for p in _walk_files():
            matches = _scan_file_for_patterns(p, SCAN_PATTERNS)
            if not matches:
                continue
            for m in matches:
                rec = {
                    "created": NOW_ISO,
                    "file": _rel(p),
                    "pattern": m["pattern"],
                    "context": m["context"]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count_hits += 1
    print(f"[TRAP] ok — {count_hits} hits -> {out.name}")

BOOTSTRAP_CEASE = """ΔINSTANT_CEASE_ORDER — OFFLINE DRAFT
Date: {date}
Claimant: {claimant}
Repo: {repo}
Seal SHA-256: {sha}

To Whom It May Concern,

This is a formal instant cease order regarding unauthorized use, alteration, concealment, or obstruction concerning sealed works, scrolls, and glyph systems operating under the TruthLock/GodKey framework.

You are instructed to immediately CEASE AND DESIST from any further interference. This notice is recorded in the sealed ledger and will be escalated upon non-compliance.

— ΔBRAKE Offline
"""

def phase_enforce(claimant="Matthew Dewayne Porter"):
    _ensure_dirs()
    tmpl = TEMPLATES / "ΔINSTANT_CEASE_ORDER.txt"
    if not tmpl.exists():
        TEMPLATES.mkdir(parents=True, exist_ok=True)
        tmpl.write_text(BOOTSTRAP_CEASE, encoding="utf-8")

    sha = "unknown"
    seal_path = OUTDIR / "ΔORIGIN_SEAL.json"
    if seal_path.exists():
        try:
            sha = json.loads(seal_path.read_text(encoding="utf-8")).get("fingerprint_sha256", "unknown")
        except Exception:
            pass

    content = (tmpl.read_text(encoding="utf-8")
               .format(date=NOW_ISO, claimant=claimant, repo=str(ROOT), sha=sha))
    (OUTDIR / "ΔINSTANT_CEASE_ORDER.txt").write_text(content, encoding="utf-8")
    print(f"[ENFORCE] ok — ΔINSTANT_CEASE_ORDER.txt written")

def run_all(allow_hooks: bool, claimant: str):
    phase_seal()
    phase_trap()
    phase_enforce(claimant=claimant)
    phase_deploy(allow_hooks=allow_hooks)
    print("[ALL] complete")

def main():
    ap = argparse.ArgumentParser(description="ΔBRAKE offline local runner (no GitHub, no network).")
    ap.add_argument("phase", choices=["seal","deploy","trap","enforce","all"], help="Which phase to run")
    ap.add_argument("--allow-hooks", action="store_true", help="Allow executing scripts in ./hooks/* (off by default)")
    ap.add_argument("--claimant", default="Matthew Dewayne Porter", help="Name to place on ΔINSTANT_CEASE_ORDER")
    args = ap.parse_args()

    if args.phase == "seal":
        phase_seal()
    elif args.phase == "deploy":
        phase_deploy(allow_hooks=args.allow_hooks)
    elif args.phase == "trap":
        phase_trap()
    elif args.phase == "enforce":
        phase_enforce(claimant=args.claimant)
    elif args.phase == "all":
        run_all(allow_hooks=args.allow_hooks, claimant=args.claimant)

if __name__ == "__main__":
    main()
[![Discord Follow](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2Fautogpt%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&label=total%20members&logo=discord&logoColor=white&color=7289da)](https://discord.gg/autogpt) &ensp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Auto_GPT?style=social)](https://twitter.com/Auto_GPT) &ensp;

**AutoGPT** is a powerful platform that allows you to create, deploy, and manage continuous AI agents that automate complex workflows. 

## Hosting Options 
   - Download to self-host (Free!)
   - [Join the Waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta (Closed Beta - Public release Coming Soon!)

## How to Self-Host the AutoGPT Platform
> [!NOTE]
> Setting up and hosting the AutoGPT Platform yourself is a technical process. 
> If you'd rather something that just works, we recommend [joining the waitlist](https://bit.ly/3ZDijAI) for the cloud-hosted beta.

### System Requirements

Before proceeding with the installation, ensure your system meets the following requirements:

#### Hardware Requirements
- CPU: 4+ cores recommended
- RAM: Minimum 8GB, 16GB recommended
- Storage: At least 10GB of free space

#### Software Requirements
- Operating Systems:
  - Linux (Ubuntu 20.04 or newer recommended)
  - macOS (10.15 or newer)
  - Windows 10/11 with WSL2
- Required Software (with minimum versions):
  - Docker Engine (20.10.0 or newer)
  - Docker Compose (2.0.0 or newer)
  - Git (2.30 or newer)
  - Node.js (16.x or newer)
  - npm (8.x or newer)
  - VSCode (1.60 or newer) or any modern code editor

#### Network Requirements
- Stable internet connection
- Access to required ports (will be configured in Docker)
- Ability to make outbound HTTPS connections

### Updated Setup Instructions:
We've moved to a fully maintained and regularly updated documentation site.

👉 [Follow the official self-hosting guide here](https://docs.agpt.co/platform/getting-started/)


This tutorial assumes you have Docker, VSCode, git and npm installed.

---

#### ⚡ Quick Setup with One-Line Script (Recommended for Local Hosting)

Skip the manual steps and get started in minutes using our automatic setup script.

For macOS/Linux:
```
curl -fsSL https://setup.agpt.co/install.sh -o install.sh && bash install.sh
```

For Windows (PowerShell):
```
powershell -c "iwr https://setup.agpt.co/install.bat -o install.bat; ./install.bat"
```

This will install dependencies, configure Docker, and launch your local instance — all in one go.

### 🧱 AutoGPT Frontend

The AutoGPT frontend is where users interact with our powerful AI automation platform. It offers multiple ways to engage with and leverage our AI agents. This is the interface where you'll bring your AI automation ideas to life:

   **Agent Builder:** For those who want to customize, our intuitive, low-code interface allows you to design and configure your own AI agents. 
   
   **Workflow Management:** Build, modify, and optimize your automation workflows with ease. You build your agent by connecting blocks, where each block     performs a single action.
   
   **Deployment Controls:** Manage the lifecycle of your agents, from testing to production.
   
   **Ready-to-Use Agents:** Don't want to build? Simply select from our library of pre-configured agents and put them to work immediately.
   
   **Agent Interaction:** Whether you've built your own or are using pre-configured agents, easily run and interact with them through our user-friendly      interface.

   **Monitoring and Analytics:** Keep track of your agents' performance and gain insights to continually improve your automation processes.

[Read this guide](https://docs.agpt.co/platform/new_blocks/) to learn how to build your own custom blocks.

### 💽 AutoGPT Server

The AutoGPT Server is the powerhouse of our platform This is where your agents run. Once deployed, agents can be triggered by external sources and can operate continuously. It contains all the essential components that make AutoGPT run smoothly.

   **Source Code:** The core logic that drives our agents and automation processes.
   
   **Infrastructure:** Robust systems that ensure reliable and scalable performance.
   
   **Marketplace:** A comprehensive marketplace where you can find and deploy a wide range of pre-built agents.

### 🐙 Example Agents

Here are two examples of what you can do with AutoGPT:

1. **Generate Viral Videos from Trending Topics**
   - This agent reads topics on Reddit.
   - It identifies trending topics.
   - It then automatically creates a short-form video based on the content. 

2. **Identify Top Quotes from Videos for Social Media**
   - This agent subscribes to your YouTube channel.
   - When you post a new video, it transcribes it.
   - It uses AI to identify the most impactful quotes to generate a summary.
   - Then, it writes a post to automatically publish to your social media. 

These examples show just a glimpse of what you can achieve with AutoGPT! You can create customized workflows to build agents for any use case.

---

### **License Overview:**

🛡️ **Polyform Shield License:**
All code and content within the `autogpt_platform` folder is licensed under the Polyform Shield License. This new project is our in-developlemt platform for building, deploying and managing agents.</br>_[Read more about this effort](https://agpt.co/blog/introducing-the-autogpt-platform)_

🦉 **MIT License:**
All other portions of the AutoGPT repository (i.e., everything outside the `autogpt_platform` folder) are licensed under the MIT License. This includes the original stand-alone AutoGPT Agent, along with projects such as [Forge](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge), [agbenchmark](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) and the [AutoGPT Classic GUI](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend).</br>We also publish additional work under the MIT Licence in other repositories, such as [GravitasML](https://github.com/Significant-Gravitas/gravitasml) which is developed for and used in the AutoGPT Platform. See also our MIT Licenced [Code Ability](https://github.com/Significant-Gravitas/AutoGPT-Code-Ability) project.

---
### Mission
Our mission is to provide the tools, so that you can focus on what matters:

- 🏗️ **Building** - Lay the foundation for something amazing.
- 🧪 **Testing** - Fine-tune your agent to perfection.
- 🤝 **Delegating** - Let AI work for you, and have your ideas come to life.

Be part of the revolution! **AutoGPT** is here to stay, at the forefront of AI innovation.

**📖 [Documentation](https://docs.agpt.co)**
&ensp;|&ensp;
**🚀 [Contributing](CONTRIBUTING.md)**

---
## 🤖 AutoGPT Classic
> Below is information about the classic version of AutoGPT.

**🛠️ [Build your own Agent - Quickstart](classic/FORGE-QUICKSTART.md)**

### 🏗️ Forge

**Forge your own agent!** &ndash; Forge is a ready-to-go toolkit to build your own agent application. It handles most of the boilerplate code, letting you channel all your creativity into the things that set *your* agent apart. All tutorials are located [here](https://medium.com/@aiedge/autogpt-forge-e3de53cc58ec). Components from [`forge`](/classic/forge/) can also be used individually to speed up development and reduce boilerplate in your agent project.

🚀 [**Getting Started with Forge**](https://github.com/Significant-Gravitas/AutoGPT/blob/master/classic/forge/tutorials/001_getting_started.md) &ndash;
This guide will walk you through the process of creating your own agent and using the benchmark and user interface.

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/forge) about Forge

### 🎯 Benchmark

**Measure your agent's performance!** The `agbenchmark` can be used with any agent that supports the agent protocol, and the integration with the project's [CLI] makes it even easier to use with AutoGPT and forge-based agents. The benchmark offers a stringent testing environment. Our framework allows for autonomous, objective performance evaluations, ensuring your agents are primed for real-world action.

<!-- TODO: insert visual demonstrating the benchmark -->

📦 [`agbenchmark`](https://pypi.org/project/agbenchmark/) on Pypi
&ensp;|&ensp;
📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/benchmark) about the Benchmark

### 💻 UI

**Makes agents easy to use!** The `frontend` gives you a user-friendly interface to control and monitor your agents. It connects to agents through the [agent protocol](#-agent-protocol), ensuring compatibility with many agents from both inside and outside of our ecosystem.

<!-- TODO: insert screenshot of front end -->

The frontend works out-of-the-box with all agents in the repo. Just use the [CLI] to run your agent of choice!

📘 [Learn More](https://github.com/Significant-Gravitas/AutoGPT/tree/master/classic/frontend) about the Frontend

### ⌨️ CLI

[CLI]: #-cli

To make it as easy as possible to use all of the tools offered by the repository, a CLI is included at the root of the repo:

```shell
$ ./run
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agent      Commands to create, start and stop agents
  benchmark  Commands to start the benchmark and list tests and categories
  setup      Installs dependencies needed for your system.
```

Just clone the repo, install dependencies with `./run setup`, and you should be good to go!

## 🤔 Questions? Problems? Suggestions?

### Get help - [Discord 💬](https://discord.gg/autogpt)

[![Join us on Discord](https://invidget.switchblade.xyz/autogpt)](https://discord.gg/autogpt)

To report a bug or request a feature, create a [GitHub Issue](https://github.com/Significant-Gravitas/AutoGPT/issues/new/choose). Please ensure someone else hasn't created an issue for the same topic.

## 🤝 Sister projects

### 🔄 Agent Protocol

To maintain a uniform standard and ensure seamless compatibility with many current and future applications, AutoGPT employs the [agent protocol](https://agentprotocol.ai/) standard by the AI Engineer Foundation. This standardizes the communication pathways from your agent to the frontend and benchmark.

---

## Stars stats

<p align="center">
<a href="https://star-history.com/#Significant-Gravitas/AutoGPT">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Significant-Gravitas/AutoGPT&type=Date" />
  </picture>
</a>
</p>


## ⚡ Contributors

<a href="https://github.com/Significant-Gravitas/AutoGPT/graphs/contributors" alt="View Contributors">
  <img src="https://contrib.rocks/image?repo=Significant-Gravitas/AutoGPT&max=1000&columns=10" alt="Contributors" />
</a>
