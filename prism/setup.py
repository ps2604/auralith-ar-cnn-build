from setuptools import setup, find_packages

setup(
    name='prism_train',
    version='1.0.0',
    description='PRISM Training Module for Auralith',
    author='Pirassena Sabaratnam Founder of Auralith',
    packages=find_packages(),
    py_modules=['prism_train'],
    install_requires=[
        'numpy',
        'opencv-python-headless',
        'Pillow',
        'google-cloud-storage',
        'tensorflow>=2.11.0',
        'matplotlib',
        'tqdm',
        'tensorflow-addons'
    ],
    entry_points={
        'console_scripts': [
            'prism_train = prism_train:main'
        ]
    },
    include_package_data=True,
    zip_safe=False,
)
