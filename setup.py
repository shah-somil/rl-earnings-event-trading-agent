"""
Setup script for EETA - Earnings Event Trading Agent.

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]
else:
    requirements = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "torch>=2.0.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
    ]

setup(
    name="eeta",
    version="2.0.0",
    author="EETA Project",
    author_email="eeta@example.com",
    description="Multi-Agent Reinforcement Learning for Earnings-Based Trading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/earnings-trading-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "demo": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
        ],
        "data": [
            "yfinance>=0.2.28",
            "finnhub-python>=2.4.18",
        ],
    },
    entry_points={
        "console_scripts": [
            "eeta=run:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
)
