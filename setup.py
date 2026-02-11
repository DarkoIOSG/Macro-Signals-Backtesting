from setuptools import setup, find_packages

setup(
    name="bitcoin-return-prediction",
    version="0.1.0",
    description="Machine learning models for Bitcoin return prediction",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)