import sys

from setuptools import setup
from setuptools import find_packages

if sys.version_info < (3, 6):
    print("Python 3.6 or higher required, please upgrade.")
    sys.exit(1)

version = "0.1"

p=find_packages('src')

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="niot",
    install_requires=required,
    description="Network Inpainting via Optimal Transport",
    version=version,
    author="Enrico Facca",
    license="MIT",
    packages=['niot'],
    package_dir={'':'src'}
)
