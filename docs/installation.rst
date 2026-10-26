.. _InstallationFile:

Installation
============

*Atmodeller* is a Python package that requires Python 3.11 or later. It is tested on Linux and macOS with Python 3.11 and 3.13. Windows is supported by JAX and should work, but it is not tested.

Python environment
------------------
It is recommended to install *Atmodeller* in a virtual environment, whether you proceed with a quick install or a full developer setup. This helps isolate dependencies and maintain reproducibility.

For more information, see the `Python documentation on venv <https://docs.python.org/3/library/venv.html>`_.

We recommend using a modern dependency manager such as `uv <https://docs.astral.sh/uv>`_, which offers fast installs and reproducible environments by default (see :ref:`InstallWithUv` below).

1. Quick install
----------------

Use pip::

    pip install atmodeller

This installs the latest release from `PyPI <https://pypi.org/project/atmodeller/>`_, which is version 2. This documentation describes version 2, whose interface differs from version 1.

Downloading the source code is also recommended if you'd like access to the example notebooks in ``notebooks/examples/``.

.. _developer_install:

2. Developer install
--------------------

2a. Fork and clone the repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to contribute to *Atmodeller* or want to work independently, it's recommended to first fork the repository to your personal account or to an organisation you belong to. This gives you full control over your changes and the ability to manage your own branches.

Follow these steps:

- Visit the main repository on GitHub: https://github.com/ExPlanetology/atmodeller
- Click the **Fork** button (usually in the top-right corner of the page).
- Choose whether to fork the repository to your personal account or to an organisation account.

After forking the repository, you should **clone your fork** (not the original) to begin development.

Cloning your fork:

- Using SSH::

    git clone git@github.com:<your-account>/atmodeller.git
    cd atmodeller

- Using HTTPS::

    git clone https://github.com/<your-account>/atmodeller.git
    cd atmodeller

Replace ``<your-account>`` with your actual **GitHub username or organisation name**. You can now work on your fork independently, create branches, and make changes as needed.

To keep your fork in sync with the original repository---or to submit changes via pull requests---you can follow the instructions in the `GitHub documentation <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork>`_ on configuring a remote upstream. This setup allows you to fetch updates from the main repository and integrate them into your fork.

To reduce the complexity of future merges, it is **strongly recommended to keep your fork's main branch closely aligned with the upstream repository**. Avoid letting your fork diverge significantly. Submit bug fix pull requests as soon as possible. New features can be submitted at any time, as long as they are self-contained and do not break any existing infrastructure. This strategy helps ensure a smoother integration process and minimises maintenance burdens.

.. note::

    You can also clone the main repository directly without forking, but this approach provides less flexibility and does not allow you to submit pull requests unless you have write access to the main repository.

2b. Install *Atmodeller*
^^^^^^^^^^^^^^^^^^^^^^^^

To install the package, use `uv <https://docs.astral.sh/uv>`_ (recommended), which installs the exact versions of the dependencies in the project's lock file (``uv.lock``), or the standard `pip <https://pip.pypa.io/en/stable/getting-started/>`_ tool, which installs the latest versions allowed by ``pyproject.toml``.

.. _InstallWithUv:

Option 1: uv (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

This requires `uv <https://docs.astral.sh/uv>`_ to be installed.

In the directory of the source code repository (where ``pyproject.toml`` is located), run::

    uv sync

``uv`` will create a virtual environment in a ``.venv`` directory inside the source repository and install the package along with its development tools and dependencies, using the versions in ``uv.lock``. These are the versions tested by the continuous integration, and ``uv sync --locked`` reproduces its environment exactly. After updating the source code (e.g. with ``git pull``), run ``uv sync`` again to bring the environment up to date.

To activate the virtual environment the command is different depending on your operating system:

Mac/Linux::

    source .venv/bin/activate

Windows (PowerShell, or ``.venv\Scripts\activate.bat`` in the Command Prompt)::

    .venv\Scripts\Activate.ps1

For all operating systems you can deactivate (exit) the virtual environment by running::

    deactivate

Optional extras:

- To install the documentation dependencies without the development tools::

      uv sync --extra docs --no-dev

- To install everything (core + dev + docs)::

      uv sync --extra docs

.. note::

    If you're using VS Code, you may need to restart the editor for the virtual environment to be detected automatically. In some cases, additional configuration may be required---see the `official guidance <https://code.visualstudio.com/docs/python/environments>`_.

    As a fallback, prefixing any command with ``uv run`` (e.g. ``uv run pytest``) will ensure it runs in the local (uv-created) environment, even if the virtual environment is not activated manually.

Option 2: pip
~~~~~~~~~~~~~

Create a virtual environment (if you haven't already), typically at the uppermost level of the source code repository. Make sure that the Python version is 3.11 or later::

    python -m venv .venv

To activate the virtual environment the command is different depending on your operating system:

Mac/Linux::

    source .venv/bin/activate

Windows (PowerShell, or ``.venv\Scripts\activate.bat`` in the Command Prompt)::

    .venv\Scripts\Activate.ps1

You may use the ``-e`` option for an `editable install <https://setuptools.pypa.io/en/latest/userguide/development_mode.html>`_::

    pip install -e .

To install additional dependencies:

- For development tools, which are a dependency group (requires pip 25.1 or later)::

      pip install -e . --group dev

- For documentation tools::

      pip install -e .[docs]

- For both::

      pip install -e .[docs] --group dev

.. note::

    Zsh treats square brackets (``[ ]``) as globbing characters. You must quote or escape them when using ``pip``. Use either of the following::

        pip install -e '.[docs]'
        # or
        pip install -e .\[docs\]

.. _DependencyNotes:

3. Dependency notes
-------------------
The dependencies in *Atmodeller* are defined in the ``pyproject.toml`` file, with special handling for JAX:

- **Apple Intel (x86_64)**: JAX is pinned to version ``0.4.38``, the last release that supports this architecture. JAX ``0.4.38`` is available for Python 3.10 to 3.13, so use Python 3.11, 3.12 or 3.13 on an Intel Mac. Dependencies that require a newer JAX (such as ``lineax``) are kept at the last compatible version.
- **All other platforms**: JAX is not pinned; any release after ``0.4.38`` can be used, except versions known to be incompatible (currently ``0.7.0`` and ``0.7.1``).

This approach balances stability and reliability on older systems with the latest features and performance improvements from JAX and related libraries on supported platforms.

When working directly with the source code, update the packages in your virtual environment after updating the source code, so they match the latest ``uv.lock`` (with ``uv sync``) or ``pyproject.toml`` (by reinstalling with ``pip``). This ensures compatibility with the current development state of *Atmodeller* and avoids conflicts caused by outdated dependencies.

If you encounter issues installing *Atmodeller*, work through the usual checks:

1. Is the Python version 3.11 or later?
2. Is the correct environment activated?
3. Are the installed packages up to date with the source code? (With uv, run ``uv sync``; see also the `uv documentation on locking and syncing <https://docs.astral.sh/uv/concepts/projects/sync/>`_.)

Error messages from the resolver are often helpful in diagnosing dependency conflicts. If you suspect the problem is specific to *Atmodeller*, please open an `issue <https://github.com/ExPlanetology/atmodeller/issues>`__ on GitHub.
