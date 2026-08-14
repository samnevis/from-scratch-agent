from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

DENIED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "ctypes",
    "pathlib",
    "shutil",
    "importlib",
    "multiprocessing",
    "threading",
    "pickle",
    "marshal",
    "code",
    "posix",
    "nt",
    "signal",
    "resource",
    "pty",
    "fcntl",
    "builtins",
}


@dataclass
class SandboxResult:
    passed: int
    total: int
    ok: bool
    stdout: str
    stderr: str
    timed_out: bool = False
    denied: str | None = None

    def summary(self, limit: int = 2000) -> str:
        if self.denied:
            return f"denied import: {self.denied}"
        if self.timed_out:
            return "timeout"
        body = f"passed {self.passed}/{self.total}"
        extra = (self.stderr or self.stdout).strip()
        if extra and not self.ok:
            body += "\n" + extra[:limit]
        return body[:limit]


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + "\n…truncated"


def find_denied_imports(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in DENIED_IMPORTS:
                    return root
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                if root in DENIED_IMPORTS:
                    return root
    return None


_RUNNER = r'''
import json, traceback
ns = {}
src = open("solution.py", encoding="utf-8").read()
try:
    exec(compile(src, "solution.py", "exec"), ns, ns)
except Exception:
    print(json.dumps({"passed": 0, "total": 0, "ok": False, "error": traceback.format_exc()}))
    raise SystemExit(0)
tests = json.loads(open("tests.json", encoding="utf-8").read())
passed = 0
errors = []
for i, stmt in enumerate(tests):
    try:
        exec(compile(stmt, f"<test{i}>", "exec"), ns, ns)
        passed += 1
    except Exception as e:
        errors.append(f"test {i}: {stmt!r} -> {type(e).__name__}: {e}")
print(json.dumps({"passed": passed, "total": len(tests), "ok": passed == len(tests) and len(tests) > 0, "errors": errors}))
'''


def run_tests(
    code: str,
    tests: list[str],
    timeout_s: float = 5.0,
    truncate_chars: int = 2000,
) -> SandboxResult:
    denied = find_denied_imports(code)
    if denied:
        return SandboxResult(0, len(tests), False, "", "", denied=denied)
    with tempfile.TemporaryDirectory(prefix="kata-sbx-") as td:
        root = Path(td)
        (root / "solution.py").write_text(code, encoding="utf-8")
        import json

        (root / "tests.json").write_text(json.dumps(tests), encoding="utf-8")
        (root / "runner.py").write_text(_RUNNER, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "runner.py"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(0, len(tests), False, "", "timeout", timed_out=True)
        stdout = _truncate(proc.stdout, truncate_chars)
        stderr = _truncate(proc.stderr, truncate_chars)
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            return SandboxResult(0, len(tests), False, stdout, stderr)
        return SandboxResult(
            passed=int(payload.get("passed", 0)),
            total=int(payload.get("total", len(tests))),
            ok=bool(payload.get("ok", False)),
            stdout=stdout,
            stderr=_truncate("\n".join(payload.get("errors", [])) or stderr, truncate_chars),
        )
