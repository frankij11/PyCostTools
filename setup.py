import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    depends = fh.read()

setuptools.setup(
    name="PyCostTools-kjoy11", # Replace with your own username
    version="0.0.1",
    author="Kevin Joy",
    author_email="kevinfjoy@gmail.com",
    description="A suite of tools for helping cost estimators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/frankij11/PyCostTools",
    packages=setuptools.find_packages(),
    install_requires = depends,
    #include_package_data = True,
    package_data={"": ['data/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
