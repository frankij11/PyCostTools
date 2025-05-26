import setuptools
import os
from pycost import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip()]

setuptools.setup(
    name="pycost",
    version=__version__,
    author="Kevin Joy",
    author_email="kevinfjoy@gmail.com",
    description="Python tools for cost estimation and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frankij11/PyCostTools",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    package_data={"pycost": ['data/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial",
    ],
    python_requires='>=3.6',
    test_suite="tests",
)
