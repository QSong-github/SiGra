import setuptools
import io
import os

def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


with open("README.md", "r") as fh:
    long_description = fh.read()

def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]

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
    install_requires=read_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
