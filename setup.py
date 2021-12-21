import pathlib
from setuptools import setup

README_PATH = pathlib.Path(__file__).parent / "README.md"
README_TEXT = README_PATH.read_text()

setup(
    name="spaceopt",
    version="0.2.1",
    license="MIT",
    description="Optimize discrete search spaces via gradient boosting regression",
    long_description=README_TEXT,
    long_description_content_type="text/markdown",
    author="Arkadiusz Nowaczyński",
    author_email="ar.nowaczynski@gmail.com",
    url="https://github.com/ar-nowaczynski/spaceopt",
    packages=["spaceopt"],
    python_requires=">=3.7",
    install_requires=[
        "lightgbm>=3.3.0",
        "pandas>=1.3.0",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
