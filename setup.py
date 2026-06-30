from setuptools import setup, find_packages

setup(
    name="rag_project",
    version="1.0",
    packages=find_packages(),
    py_modules=["config"]  # this tells pip to include config.py
)