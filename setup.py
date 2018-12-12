# -*- coding: utf-8 -*-

"""
To upload to PyPI, PyPI test, or a local server:
python setup.py bdist_wheel upload -r <server_identifier>
"""

import setuptools
import os

setuptools.setup(
    name="nionswift-eels-analysis",
    version="0.3.0",
    author="Nion Software",
    author_email="swift@nion.com",
    description="Library and UI for doing EELS analysis with Nion Swift.",
    long_description=open("README.rst").read(),
    url="https://github.com/nion-software/eels-analysis",
    packages=["nion.eels_analysis", "nion.eels_analysis.test", "nionswift_plugin.nion_eels_analysis", "nionswift_plugin.nion_eels_analysis.test"],
    package_data={"nion.eels_analysis": ["resources/*"]},
    install_requires=["nionswift>=0.14.0"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.6",
    ],
    include_package_data=True,
    test_suite="nion.eels_analysis.test",
    python_requires='~=3.5',
)
