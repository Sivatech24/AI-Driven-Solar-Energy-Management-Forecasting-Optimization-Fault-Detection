from setuptools import setup, find_packages

setup(
    name="Solar_Power_Prediction_Tensorflow_LSTM_Model",
    version="1.0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "scikit-learn"
    ],
    author="CodingMaster24",
    author_email="sivatech24@gmail.com",
    description="A package for solar power prediction using LSTM in TensorFlow",
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
