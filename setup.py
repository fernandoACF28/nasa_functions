from setuptools import setup, find_packages

setup(
    name="funcitons_nasa",
    version="0.2",
    packages=find_packages(),
    install_requires=[],  # Dependências (adicione aqui se precisar)
    author="Fernando Fernandes",
    author_email="fernando.allysson@usp.br",
    description="This is packpage with utils functions for makedownload and extract temporal series from hdf_files",
    url="https://github.com/fernandoACF28/nasa_functions",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
