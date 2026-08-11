from setuptools import setup, find_packages

setup(
    name='acadia_qmsmt',
    version='1.0',
    packages=find_packages(),  # Auto-detects all submodules and subpackages
    install_requires=[
        'lmfit',
        'uncertainties',
        'ruamel.yaml'
    ]
)