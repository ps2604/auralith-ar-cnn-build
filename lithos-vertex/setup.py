from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='lithostrain',
    version='0.1',
    py_modules=['lithostrain'],
    install_requires=requirements,
)
