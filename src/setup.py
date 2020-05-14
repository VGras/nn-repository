import os
import setuptools


setuptools.setup(
    name                            = "orquestra-nn",
    packages                        = setuptools.find_packages(where = "python"),
    package_dir                     = {"" : "python"},
    classifiers                     = (
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    )
)
