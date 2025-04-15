from setuptools import setup

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='auralith_fluxa_training',
    version='0.1',
    py_modules=['train'],
    install_requires=requirements,
)
