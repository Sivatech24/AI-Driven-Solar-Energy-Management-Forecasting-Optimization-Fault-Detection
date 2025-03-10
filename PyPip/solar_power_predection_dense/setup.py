from setuptools import setup, find_packages

setup(
    name="Solar_Power_Prediction_Using_Tensorflow_Dense_Model",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn"
    ],
    author="CodingMaster24",
    author_email="sivatech24@gmail.com",
    description="A package for solar power prediction using Dense in TensorFlow",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sivatech24/AI-Driven-Solar-Energy-Management-Forecasting-Optimization-Fault-Detection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
