from skbuild import setup  # This line replaces 'from setuptools import setup'
setup(
    name="llamppl",
    version="0.0.1",
    description="LLaMPPL: probabilistic programming with language models",
    author='Alex Lew',
    packages=['llamppl'],
    package_dir={'llamppl': 'llamppl'},
    python_requires=">=3.8",
    setup_requires=['numpy'],
    install_requires=['numpy'],
)
