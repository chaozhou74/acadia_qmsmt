from distutils.core import setup

setup(
      name='acadia_qmsmt',
      version='0.0.1',
      packages=['acadia_qmsmt'],
      
      extras_require={
        "host": [
            "ruamel.yaml"
        ],
    },
)