from skbuild import setup  # This line replaces 'from setuptools import setup'
setup(
    name="lampl",
    version="0.0.1",
    description="LaMPL: probabilistic programming with language models",
    author='Alex Lew',
    packages=['lampl'],
    package_dir={'lampl': 'lampl'},
    python_requires=">=3.8",
)
