"""Minimal pytest shim providing only pytest.raises, for environments without
pytest installed. The real pytest.raises supports much more; this covers
exactly what this project's tests use.
"""
import re as _re


class _RaisesContext:
    def __init__(self, expected_exception, match=None):
        self.expected_exception = expected_exception
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.expected_exception}")
        if not issubclass(exc_type, self.expected_exception):
            return False
        if self.match and not _re.search(self.match, str(exc_val)):
            raise AssertionError(
                f"Exception message {exc_val!r} did not match pattern {self.match!r}"
            )
        return True


def raises(expected_exception, match=None):
    return _RaisesContext(expected_exception, match)
