from setuptools import setup

setup(
    name='ivhd',
    version='1.0',
    description='Implementation of IVHD method using PyTorch library',
    author='Janusz Jakubiec, Paweł Świder, Piotr Rzeźnik',
    author_email='swiderpawel51@gmail.com',
    packages=['ivhd'],
    install_requires=[
        'torch',
    ],
)
