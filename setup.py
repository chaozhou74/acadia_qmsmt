from setuptools import setup, find_packages

setup(
    name='acadia_qmsmt',
    version='0.0.1',
    packages=find_packages(),  # Auto-detects all submodules and subpackages
    extras_require={
        "host": [
            "ruamel.yaml"
        ],
    },
)