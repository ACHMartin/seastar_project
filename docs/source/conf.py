# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
autodoc_mock_imports = ["numpy", "xarray", "scipy", "utm", "pytest", "cartopy",
                        "shapely"]

sys.path.insert(0, os.path.abspath('../..'))
print(os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'SeaSTAR'
copyright = '2023, Adrien Martin, David McCann, Eva Le Merle'
author = 'Adrien Martin, David McCann, Eva Le Merle'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]

# -- Options for LaTeX output ---------------------------------------------
latex_engine = 'pdflatex'
latex_elements = {
    'babel': '\\usepackage[shorthands=off]{babel}',
    'preamble': r'''
    \usepackage{makeidx}
    \usepackage[columns=2]{idxlayout}
    \makeindex
    '''
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

# Theme options
html_theme_options = {
    'collapse_navigation': False,  # Garde le menu de navigation ouvert
    'sticky_navigation': True,     # Garde la navigation en vue lorsque vous descendez sur la page
    'navigation_depth': 2,         # Profondeur maximale de la hiérarchie dans la navigation
    'titles_only': False,          # Afficher uniquement les titres dans la navigation latérale
}
