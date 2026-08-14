from sandbox.runner import find_denied_imports, run_tests


def test_run_tests_pass():
    code = "def add(a, b):\n    return a + b\n"
    r = run_tests(code, ["assert add(1, 2) == 3"])
    assert r.ok and r.passed == 1


def test_run_tests_fail():
    code = "def add(a, b):\n    return a - b\n"
    r = run_tests(code, ["assert add(1, 2) == 3"])
    assert not r.ok


def test_denied_import():
    code = "import os\ndef add(a, b):\n    return a + b\n"
    assert find_denied_imports(code) == "os"
    r = run_tests(code, ["assert add(1, 2) == 3"])
    assert not r.ok and r.denied == "os"
