# Publishing Your Python Package to PyPI

This guide provides a step-by-step process for publishing your Python package to the Python Package Index (PyPI).

## 1. Introduction to PyPI

PyPI (Python Package Index) is the official third-party software repository for Python. It allows developers to share their Python packages with the wider Python community, making it easy for others to install and use them.

## 2. Prerequisites

Before you can publish your package, ensure you have the following:

*   **A PyPI Account**: You'll need an account on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/). TestPyPI is a separate instance of PyPI for testing the packaging and distribution process.
*   **Python Installed**: A recent version of Python should be installed on your system.
*   **`pip` Up to Date**: Ensure your `pip` (Python package installer) is up to date:
    ```bash
    python -m pip install --upgrade pip
    ```

## 3. Preparing Your Package for Publication

Your package needs several files and configurations to be ready for PyPI.

*   **`setup.py` (or `pyproject.toml`)**:
    This file is the build script for your package. It tells `setuptools` (and other build tools) about your package (such as the name and version) as well as which code files to include.
    Ensure the following fields are accurately filled:
    *   `name`: The name of your package (e.g., `pycost`).
    *   `version`: The version of your package. Follow semantic versioning (`MAJOR.MINOR.PATCH`).
        *   `MAJOR` version when you make incompatible API changes,
        *   `MINOR` version when you add functionality in a backward-compatible manner, and
        *   `PATCH` version when you make backward-compatible bug fixes.
        *   It's crucial to update the version in `pycost/__init__.py` (if your `setup.py` reads it from there) before each release.
    *   `author`: Your name or your organization's name.
    *   `description`: A short, one-sentence summary of the package.
    *   `long_description`: A detailed description of the package. This is often taken from your `README.md`.
    *   `long_description_content_type`: Set this to `text/markdown` if your `long_description` is in Markdown.
    *   `url`: The URL for the homepage of your project (e.g., GitHub repository).
    *   `packages`: A list of all Python import packages that should be included in your distribution package. `setuptools.find_packages()` can find these automatically.
    *   `install_requires`: A list of other packages that your package depends on to run.
    *   `python_requires`: The version of Python your package is compatible with (e.g., `>=3.7`).
    *   `classifiers`: A list of classifiers to categorize your project (e.g., `Programming Language :: Python :: 3`, `License :: OSI Approved :: MIT License`, `Operating System :: OS Independent`).

*   **`README.md`**:
    This file provides a good overview of your package. It should be up-to-date and clearly explain what your package does, how to install it, and how to use it. This is often used as the `long_description` for PyPI.

*   **`LICENSE.txt` (or `LICENSE`)**:
    This file must be present and contain the full text of the license under which your software is distributed (e.g., MIT License, Apache License 2.0).

*   **`CHANGES.md` or `NEWS.md` (Recommended)**:
    It's good practice to maintain a file that tracks changes, new features, and bug fixes for each version of your package.

*   **`.gitignore`**:
    Ensure that build artifacts and other non-source files are ignored by Git. Add these to your `.gitignore`:
    ```
    # Build artifacts
    dist/
    build/
    *.egg-info/

    # Other
    __pycache__/
    *.pyc
    .pytest_cache/
    ```

## 4. Building Your Package

To build your package, you'll need some tools.

*   **Install Build Tools**:
    ```bash
    pip install build twine wheel
    ```
    *   `build`: The primary tool for building Python packages.
    *   `twine`: For uploading your package to PyPI.
    *   `wheel`: For creating wheel distributions (often a dependency of `build`).

*   **Build Command**:
    Navigate to the root directory of your package (where `setup.py` or `pyproject.toml` is located) and run:
    ```bash
    python -m build
    ```

*   **Distribution Formats**:
    This command will create two types of distribution files in a new `dist/` directory:
    *   **`sdist` (Source Distribution)**: A `.tar.gz` file containing your source code and the `setup.py` script. This allows users to build the package on their system.
    *   **`bdist_wheel` (Built Distribution)**: A `.whl` file. Wheels are pre-built packages that install faster because they don't require a build step on the user's machine.

## 5. Testing Your Package on TestPyPI

Before publishing to the real PyPI, it's highly recommended to test the process using TestPyPI.

*   **Purpose of TestPyPI**: TestPyPI is a separate PyPI instance for testing. Packages uploaded here are not permanent and are not indexed by `pip` by default.

*   **Upload to TestPyPI**:
    ```bash
    twine upload --repository testpypi dist/*
    ```
    You will be prompted for your TestPyPI username and password. Use the token you created on TestPyPI as the password.

*   **Install from TestPyPI**:
    To verify that your package was uploaded correctly and can be installed, create a new virtual environment and try installing it:
    ```bash
    pip install --index-url https://test.pypi.org/simple/ --no-deps pycost
    ```
    (Replace `pycost` with your actual package name if different). The `--no-deps` flag ensures that `pip` only installs your package and not its dependencies from TestPyPI (which might not be there or might be outdated).

*   **Verify**:
    After installation, try importing your package and testing some basic functionality to ensure everything works as expected.

## 6. Publishing Your Package to PyPI

Once you are confident that your package is ready and works correctly when installed from TestPyPI, you can publish it to the official PyPI.

*   **Upload to PyPI**:
    ```bash
    twine upload dist/*
    ```
    You will be prompted for your PyPI username and password. Again, it's recommended to use an API token as the password.

    **Important**: Once a specific version of a package (e.g., `pycost-0.1.0`) is uploaded to PyPI, you cannot upload the same version again, even if you delete it from PyPI. You will need to increment the version number for any changes or re-uploads.

## 7. Post-Publication

After successfully publishing your package:

*   **Git Tagging**:
    It's a good practice to create a Git tag for the released version. This makes it easy to check out the code for a specific release.
    ```bash
    git tag v0.1.0  # Replace 0.1.0 with your released version
    git push origin v0.1.0 # Push the tag to the remote repository
    # Or push all tags:
    # git push --tags
    ```

*   **Announcing the Release (Optional)**:
    You might want to announce your new release on relevant platforms (e.g., project mailing list, social media, etc.).

This guide should help you get your Python package published and available to the world!
