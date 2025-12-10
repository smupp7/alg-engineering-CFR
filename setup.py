"""
Setup script for alg-engineering-CFR
"""
from setuptools import setup, find_packages

setup(
    name="alg-engineering-cfr",
    version="0.1.0",
    description="Adaptive Payoff Matrix Sparsification for Parallel CFR",
    author="Your Name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'joblib>=1.0.0',
        'psutil>=5.8.0',
        'open_spiel>=1.0.0',
    ],
    python_requires='>=3.8',
)