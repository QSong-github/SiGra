import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SiGra",
    version="0.0.1",
    author="Ziyang Tang",
    author_email="tang385@purdue.edu",
    description="Image guided spatial transcriptomics clustering and denoising using graph transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QSong-github/SiGra",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)