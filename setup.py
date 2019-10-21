from setuptools import setup, find_packages

setup(
    name="cherab-metis",
    version="1.0.0",
    license="EUPL 1.1",
    namespace_packages=['cherab'],
    packages=find_packages(),
    include_package_data=True,
)

