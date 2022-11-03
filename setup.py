"""Install packages as defined in this file into the Python environment."""
from setuptools import setup, find_packages

import re

# The version of this tool is based on the following steps:
# https://packaging.python.org/guides/single-sourcing-package-version/
VERSION = {}
REQUIREMENTS = [
    re.sub(r"(git\+.*egg=(.*))", r"\2 @ \1", i.strip())
    for i in open("requirements.txt").readlines()
]

with open("./df_trade_simulator/__init__.py") as fp:
    content = fp.read()
    split_imports = content.split("from")
    constants = split_imports[0]
    # pylint: disable=W0122
    exec(constants, VERSION)

setup(
    name="df_trade_simulator",
    author="Martin Trapp",
    author_email="info@martintrapp.com",
    description="Simulates trading strategies through out a Pandas data frame.",
    version=VERSION.get("__version__", "0.0.0"),
    packages=find_packages(where=".", exclude=["tests"]),
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3.0",
        "Topic :: Utilities",
        "Development Status :: 3 - Alpha",
    ],
)
