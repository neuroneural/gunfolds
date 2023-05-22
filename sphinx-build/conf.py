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
                
sys.path.insert(0,os.path.abspath("../../"))
sys.path.insert(0,os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = ''
copyright = 'neuroneural.net'
author = 'neuroneural.net'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx_copybutton",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme_options = {
    "light_logo": "light.png",
    "dark_logo": "dark.png",
    "source_edit_link": "https://github.com/neuroneural/gunfolds/tree/master/sphinx-build/{filename}",
    "light_css_variables": {
        "font-stack": "FreightSans, Helvetica Neue, Helvetica, Arial, sans-serif",
        "font-family": "FreightSans, Helvetica Neue, Helvetica, Arial, sans-serif",
        "color-api-overall":"green",
        "color-api-pre-name":"#005eff",
    },
}
html_title = ''
html_theme = 'furo'
html_favicon = '../gunfolds/gallery/favicon.png'
html_static_path = ['../gunfolds/gallery']
pygments_style = "sphinx"
