from setuptools import setup, find_packages

setup(
    name='retnet-torch',
    packages = find_packages(exclude=['tests',]),
    version = '0.0.1',
    license='MIT',
    description = 'Strongly typed implementation of the Paper "Retentive Network: A Successor to Transformer for Large Language Models" in Pytorch.',
    author = 'juvi21',
    author_email="juv121@skiff.com",
    long_description=open('README.md').read(),
    keywords = ['retnet', 
                'retention', 
                'pytorch'],
    install_requires=[
        'torch>=1.8.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)