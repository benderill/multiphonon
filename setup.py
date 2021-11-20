from setuptools import setup, find_packages

with open("README.md",'rb') as fh:
    long_description = fh.read()

setup(
    name="multipcc",
    version="0.0.1",
    description = ("Calculate multiphonon capture coefficients"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
)
