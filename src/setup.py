from setuptools import setup, find_packages

setup(
    name="crypto-statarb",
    version="0.1.3",
    author="mar-iov",
    author_email="-",
    description="Statistical arbitrage framework for cryptocurrency pairs trading",
    url="https://github.com/mar-iov/crypto-statarb",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3.12.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
)