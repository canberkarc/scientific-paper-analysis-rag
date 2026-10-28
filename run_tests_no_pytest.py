"""Zero-dependency runner for the pytest-style test files, for use only where
pytest cannot be installed (e.g. this sandbox). On the cluster, just run
`pytest tests/` normally — this file is a verification convenience, not part
of the delivered test suite's intended entry point.
"""
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import _pytest_shim  # noqa: E402
sys.modules["pytest"] = _pytest_shim


def discover_test_files():
    return sorted((ROOT / "tests" / "unit").glob("test_*.py")) + \
           sorted((ROOT / "tests" / "integration").glob("test_*.py"))


def run():
    total = 0
    failed = 0
    for path in discover_test_files():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        test_names = [name for name in dir(module) if name.startswith("test_")]
        print(f"\n{path.relative_to(ROOT)} ({len(test_names)} tests)")
        for name in test_names:
            total += 1
            func = getattr(module, name)
            try:
                func()
                print(f"  PASS  {name}")
            except Exception as error:
                failed += 1
                print(f"  FAIL  {name}: {error}")
                traceback.print_exc(limit=3)
    print(f"\n{'=' * 60}\n{total - failed}/{total} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
