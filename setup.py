from setuptools import find_packages
from setuptools import setup

with open('README.md') as f:
    README = f.read()
    
setup(
    name='DICG',
    version='0.0.0',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[],
    license='MIT',
    long_description=README,
    long_description_content_type='text/markdown',
)
