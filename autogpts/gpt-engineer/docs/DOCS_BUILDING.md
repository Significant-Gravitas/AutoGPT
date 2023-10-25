Building Docs with Sphinx
=========================

This example shows a basic Sphinx docs project with Read the Docs. This project is using `sphinx` with `readthedocs`
project template.

Some useful links are given below to lear and contribute in the project.

📚 [docs/](https://www.sphinx-doc.org/en/master/usage/quickstart.html)<br>
A basic Sphinx project lives in `docs/`, it was generated using Sphinx defaults. All the `*.rst` & `*.md` make up sections in the documentation. Both `.rst` and `.md` formats are supported in this project

⚙️ [.readthedocs.yaml](https://docs.readthedocs.io/en/stable/config-file/v2.html)<br>
Read the Docs Build configuration is stored in `.readthedocs.yaml`.


📍 [docs/requirements.txt](https://docs.readthedocs.io/en/stable/config-file/v2.html)<br>
Python dependencies are [pinned](https://docs.readthedocs.io/en/latest/guides/reproducible-builds.html) (uses [pip-tools](https://pip-tools.readthedocs.io/en/latest/)) here. Make sure to add your Python dependencies to `requirements.txt` or if you choose [pip-tools](https://pip-tools.readthedocs.io/en/latest/), edit `docs/requirements.txt`.



Example Project usage
---------------------

`Poetry` is the package manager for `gpt-engineer`. In order to build documentation, we have to add docs requirements in
development environment.

This project has a standard readthedocs layout which is built by Read the Docs almost the same way that you would build it
locally (on your own laptop!).

You can build and view this documentation project locally - we recommend that you activate a `poetry` or your choice of `venv`
and dependency management tool.

Update `repository_stats.md` file under `docs/intro`

```console
# Install required Python dependencies (MkDocs etc.)
pip install -e .[doc]
cd docs/
# Create the `api_reference.rst`
python create_api_rst.py

# Build the docs
make html
```

Project Docs Structure
----------------------
If you are new to Read the Docs, you may want to refer to the [Read the Docs User documentation](https://docs.readthedocs.io/).

Below is the rundown of documentation structure for `pandasai`, you need to know:

1. place your `docs/` folder alongside your Python project.
2. copy `.readthedocs.yaml` and the `docs/` folder into your project root.
3. `docs/api_reference.rst` contains the API documentation created using `docstring`.  Run the `create_api_rst.py` to update the API reference file.
4. Project is using standard Google Docstring Style.
5. Rebuild the documenation locally to see that it works.
6. Documentation are hosted on [Read the Docs tutorial](https://docs.readthedocs.io/en/stable/tutorial/)


Read the Docs tutorial
----------------------

To get started with Read the Docs, you may also refer to the
[Read the Docs tutorial](https://docs.readthedocs.io/en/stable/tutorial/). I

With every release, build the documentation manually.
