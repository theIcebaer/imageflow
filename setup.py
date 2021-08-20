from setuptools import setup

setup(
    name='jarvis',
    version='0.1',
    description='Thesis project on image registration via invertible neural networks.',
    python_requires='>=3.6',
    install_requires = [
        'numpy',
        'matplotlib',
        'h5py',
        'torch',
        'FrEIA==0.2',
        'voxelmorph',
    ]

)