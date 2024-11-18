from distutils.core import setup

setup(
      name='linc_rfsoc',
      version='0.0.1',
      packages=['linc_rfsoc'],
      
      extras_require={
        "host": [
            "ruamel.yaml"
        ],
    },
)