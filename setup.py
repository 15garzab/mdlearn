from setuptools import setup, find_packages

setup(
    name='mdlearn'
    version='0.1.0'
    packages=find_packages(include=['mdlearn'])
    install_requires=['numpy, pandas, matplotlib, ovito, tensorflow']
)