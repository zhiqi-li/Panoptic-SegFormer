from distutils.core import setup
from setuptools import find_packages

import os
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] # Examples: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='easymd',
    version='0.1',
    packages=find_packages(),
    url='',
    license='Apache',
    author='Li Zhiqi',
    install_requires=install_requires,
    author_email='lzq@smail.nju.edu.cn',
    description='This package aims to enrich the ability of MMdetection'
)
