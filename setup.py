import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="pybiosci",
    version="0.0.5",
    author="Md. Mushahidul Islam Shamim",
    author_email="mushahidshamim@gmail.com",
    description="Python Bioscience package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mushahid2521/BioDiversity_Index",
    project_urls={
        "Bug Tracker": "https://github.com/Mushahid2521/BioDiversity_Index/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas>=0.23.3"
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6"
)
