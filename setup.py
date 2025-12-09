"""
Setup script for Amharic Sentiment Analysis package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="amharic-sentiment",
    version="1.0.0",
    author="Fikerte Shalemo",
    author_email="",
    description="Deep learning-based sentiment analysis for Amharic text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Fikerdeep/Amharic-Sentiment-Analysis",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "amharic-sentiment=amharic_sentiment.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
