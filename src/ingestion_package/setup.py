"""
Setup script for Medical Document Ingestion Package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Medical Document Ingestion Package with enhanced citation tracking"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        "docling>=1.0.0",
        "langchain>=0.1.0",
        "langchain-google-genai>=1.0.0",
        "sentence-transformers>=2.0.0",
        "pinecone-client>=3.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.64.0",
    ]

setup(
    name="medical-document-ingestion",
    version="1.0.0",
    author="Thera-RAG Team",
    author_email="support@thera-rag.com",
    description="Medical document ingestion with enhanced citation tracking",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/thera-rag/medical-document-ingestion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Markup",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "medical-ingest=ingestion_package.cli:main",
        ],
    },
    keywords="medical documents pdf processing citations rag vector-database nlp",
    project_urls={
        "Bug Reports": "https://github.com/thera-rag/medical-document-ingestion/issues",
        "Documentation": "https://medical-document-ingestion.readthedocs.io/",
        "Source": "https://github.com/thera-rag/medical-document-ingestion",
    },
    include_package_data=True,
    package_data={
        "ingestion_package": [
            "*.md",
            "*.txt",
            "*.yaml",
            "*.json"
        ]
    },
)