"""
Setup script for CRF (Cellular Reasoning Fabric) package
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().splitlines() 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="crf-reasoning",
    version="0.1.0",
    author="CRF Research Team",
    author_email="your-email@example.com",
    description="Cellular Reasoning Fabric: Adaptive computation via dynamic cell populations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pc-ai",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/pc-ai/issues",
        "Source": "https://github.com/yourusername/pc-ai",
        "Documentation": "https://github.com/yourusername/pc-ai/blob/main/README.md",
    },
    package_dir={"": "src", "scripts": "scripts"},
    packages=find_packages(where="src", exclude=["tests", "tests.*", "results", "results.*"]) + ["scripts"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.3.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crf-benchmark=scripts.benchmark:main",
            "crf-plot=scripts.plot:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine-learning deep-learning neural-networks transformer reasoning",
)