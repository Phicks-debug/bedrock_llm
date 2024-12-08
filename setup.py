from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="bedrock_llm",
    version="1.0.0",
    packages=find_packages(where="src"),
    include_package_data=True,
    package_dir={"": "src"},
    install_requires=requirements,
    author="Tran Quy An",
    author_email="an.tq@techxcorp.com",
    description="A Python LLM framework for interacting with AWS Bedrock services, it supports true async to interact with AWS bedrock Service for invoke model API, build on top of aiohttp. It also have synchronous version which built on top of boto3. This library serves as a comprehensive tool for fast prototyping, building POCs, and deploying production-ready LLM applications with robust infrastructure support.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Phicks-debug/bedrock_llm",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9",
    keywords="aws bedrock llm machine-learning ai",
)
