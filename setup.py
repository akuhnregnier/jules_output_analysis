# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="jules_output_analysis",
    author="Alexander Kuhn-Regnier",
    author_email="ahf.kuhnregnier@gmail.com",
    description="Analysis of JULES data.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/akuhnregnier/jules_output_analysis",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    setup_requires=["setuptools-scm"],
    use_scm_version=dict(write_to="src/jules_output_analysis/_version.py"),
    include_package_data=True,
)
