import os
import sys
sys.path.insert(0, os.path.abspath('./../../'))
import gamornet

# -- Project information -----------------------------------------------------

project = 'gamornet'
copyright = gamornet.__copyright__
author = gamornet.__author__

# The full version, including alpha/beta/rc tags
version = gamornet.__version__
release = gamornet.__version__

rst_epilog = '.. |copyright| replace:: %s' % copyright


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['recommonmark']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'default'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

### Things that need to be set this way for RTD Integration ##
master_doc = 'index'
