import datetime

current_year = datetime.datetime.now().year

project = "ivhd"
project_copyright = f"2024 - {current_year} (MIT License)"
author = 'Bartosz Ćwikła, Adrian Beściak, Błażej Nowicki, Michał Stefanik'
# release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.duration",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
]

autodoc_default_options = {
    "inherited-members": True,
    "members": "fit,fit_transform,transform",
}

templates_path = ['_templates']
exclude_patterns = [
    ".ipynb_checkpoints",
    ".DS_Store",
    "_build",
    "Thumbs.db",
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
