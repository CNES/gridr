from ._libgridr import (
        py_array1_replace_i8,
        py_array1_replace_u8,
        )

__all__ = [
        "py_array1_replace_i8",
        "py_array1_replace_u8",
        ]

# It is a common practice in Python packaging to keep the extension modules
# private and use Pure Python modules to wrap them.
# This allows you to have a very fine control over the public API.
