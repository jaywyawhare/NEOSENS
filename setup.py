from setuptools import setup, find_packages

setup(
    name='comparative_analysis',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow',
        'numpy',
        'matplotlib',
        'scikit-learn'
    ],
)
