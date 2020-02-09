from setuptools import setup, find_packages

setup(
    name='graph-to-graph',
    version='0.1',
    packages=find_packages(exclude=["tests"]),
    url='',
    license='MIT',
    author='Michail Kovanis',
    description='Multi-modal Graph-to-Graph Translation for Molecular Optimization',
    install_requires=[
        'numpy==1.17.4'
    ],
)
