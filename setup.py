from setuptools import setup, find_packages

setup(
    name="pulsar-search",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'astropy>=5.1',
        'numpy>=1.21.5',
        'pandas>=2.0.3',
        'pixell>=0.16.0',
    ],
    extras_require={
        'profiling': ['psutil>=5.9.2'],
        'parallel': ['mpi4py>=3.1.6'],
        'act': [
            'enlib @ git+https://github.com/amaurea/enlib.git',
            'enact @ git+https://github.com/amaurea/enact.git',
        ],
    },
)
