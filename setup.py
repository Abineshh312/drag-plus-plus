from setuptools import setup, find_packages

setup(
    name="drag-plus-plus",
    version="0.1.0",
    description="DRAG++: Adaptive Evidence-Aware RAG Distillation with Real-Time Hallucination Detection",
    author="Abinesh Haridoss",
    author_email="abinesha312@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[line.strip() for line in open("requirements.txt")],
)
