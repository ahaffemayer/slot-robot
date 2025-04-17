from setuptools import setup, find_packages

setup(
    name="slot_robot",
    version="0.1.0",
    description="A package to use the slot algorithm in meshcat.",
    author="Arthur Haffemayer",
    author_email="arthur.haffemayer@laas.fr",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)