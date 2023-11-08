import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-myown",
    version="1.1.0",
    author="ZihaoLiu",
    author_email="v-zihaoliu@microsoft.com",
    description="Pytorch-myown:a pytorch template for easy use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=[
            'nltk',
            'lightgbm',
            'numpy',
            'sklearn',
            'tqdm',
            'argparse',
        ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
