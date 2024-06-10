# setup.py

from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="linear_algebra",
    version="0.1.0",
    rust_extensions=[RustExtension("linear_algebra.linear_algebra", binding=Binding.PyO3)],
    packages=["linear_algebra"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
