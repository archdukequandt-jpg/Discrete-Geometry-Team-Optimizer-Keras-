#!/usr/bin/env python3
import os, sys, subprocess, webbrowser, time, venv, platform, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
APP = ROOT / "app.py"
VENV = ROOT / ".venv"

def run(cmd, **kw):
    print("[launcher]", " ".join(cmd))
    subprocess.check_call(cmd, **kw)

def ensure_venv():
    if not VENV.exists():
        print("[launcher] Creating venv...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    py = VENV / ("Scripts/python.exe" if os.name=="nt" else "bin/python")
    pip = VENV / ("Scripts/pip.exe" if os.name=="nt" else "bin/pip")
    return str(py), str(pip)

def ensure_deps(pip):
    req = ROOT/"requirements.txt"
    run([pip, "install", "-U", "pip"])
    run([pip, "install", "-r", str(req)])

def main():
    py, pip = ensure_venv()
    try:
        ensure_deps(pip)
    except Exception as e:
        print("[launcher] dependency install failed:", e)
        print("Try running inside the venv manually.")
    # Run streamlit
    cmd = [py, "-m", "streamlit", "run", str(APP), "--server.headless=true", "--server.port=8501"]
    p = subprocess.Popen(cmd, cwd=str(ROOT))
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass
    p.wait()

if __name__ == "__main__":
    main()
