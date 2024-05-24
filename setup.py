from setuptools import setup

with open("./README.md", "r") as f:
    long_description = f.read()

with open('./requirements.txt') as f:
    install_requires = f.read().splitlines()


setup(
    name="auto_ml_models",
    version="0.1.0", 
    author="Alireza Kiaeipour", 
    description="A package to find best machine learning model",
    long_description=long_description,
    install_requires=install_requires,
    author_email="a.kiaipoor@gmail.com",
    packages=["auto_ml_models"]
)
