SeaSTAR installation
====================

.. meta::
    :description: SeaSTAR Installation | You can install SeaSTAR from source.

This page describes the installation process for SeaSTAR tools.

Retrieving sources
------------------

Navigate to the latest release `(v2023.10.3)` on the right-hand-side of the root project page and download and unzip the source code.

You can also retrieve sources from the GitHub repository :

::

    $ git clone https://github.com/ACHMartin/seastar_project.git


How to install dependencies
---------------------------

It is possible to set up the SeaSTAR environment either with **Mamba/Conda** or with **Poetry**. 
Both methods are described below, highlighting the differences and use cases for each approach.

Using poetry
~~~~~~~~~~~~

**Poetry** is a modern **Python** dependency management and packaging tool designed to streamline the development and deployment of **Python** projects. 
**Poetry** focuses on pure Python projects and provides a simple interface for managing project dependencies, versions, and virtual environments. 
Unlike Conda, **Poetry** does not handle non-Python dependencies directly.
A *pyproject.toml* file is available and allow to install the environmenent using **Poetry**.

To download **Poetry**, use the following command:
::

    $ curl -sSL https://install.python-poetry.org | python3 -

Since **Poetry** does not handle complex dependencies like **Cartopy**, you first need to create an environment using **Mamba/Conda** and install the non-Python dependencies before using **Poetry**:
::

    $ mamba create -n seastar -c conda-forge python=3.8.16
    $ mamba activate
    $ mamba install cartopy=0.18.0

Once this environment is created you can move to the seastar_projectdirectory and launch the **Poetry** command to install the rest of the Python-only dependencies with the corresponding versions:
::

    $ cd seastar_project 
    $ poetry install

**Poetry** creates and manages virtual environments automatically. 
It uses a lock file *poetry.lock* to guarantee that the exact same package versions are used across all installations, making it more predictable.

Using Mamba or conda
~~~~~~~~~~~~~~~~~~~~
**Mamba** and **Conda** are powerful package managers widely used in the **Python** ecosystem. 
They are especially effective for managing complex dependencies, including packages that have non-Python dependencies, such as libraries written in C, Fortran, or that rely on external system libraries (e.g., Cartopy).

To create a new environment with all the required packages and activate this environment using **Mamba**, you can run:
::

    $ mamba env create -f seastar_project/env/environment.yml
    $ mamba activate seastar

Alternatively, using **Conda** (slightly slower than **Mamba** but widely used):
::

    $ conda env create -f seastar_project/env/environment.yml
    $ conda activate seastar

