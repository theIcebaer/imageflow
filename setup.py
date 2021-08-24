from setuptools import setup, find_packages

setup(
    name='imageflow',
    version='0.1',
    description='Thesis project on image registration via invertible neural networks.',
    python_requires='>=3.6',
    packages=find_packages(where='src',
                           include=['imageflow']),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'matplotlib',
        'h5py',
        'FrEIA @ git+https://github.com/VLL-HD/FrEIA.git@v0.2',
        'voxelmorph',
    ]

)