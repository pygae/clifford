Installation
============

Clifford can be installed via the standard mechanisms for Python packages, and is available through both PyPI and Anaconda.

conda
-----
If you are using the `Anaconda python distribution <https://www.anaconda.com/distribution/#download-section>`_, then you should install using the ``conda`` command.

Once you have anaconda installed you can install clifford from the conda-forge channel with the command::

    conda install -c conda-forge clifford

PyPI
----
If you are not using Anaconda, you can install ``clifford`` from PyPI using the ``pip`` command that comes with Python::

    pip3 install clifford

.. note ::

    The 1.3 release and older are not compatible with :mod:`numba` 0.50.0, and emit a
    large number of warnings on 0.49.0. To avoid these warnings, install with::

        pip3 install clifford "numba~=0.48.0" "sparse~=0.9.0"

If you want the development version of ``clifford`` you can ``pip install`` from the latest master with::

    pip3 install git+https://github.com/pygae/clifford.git@master

If you want to modify ``clifford`` itself, then you should use an editable (``-e``) installation::

    git clone https://github.com/pygae/clifford.git
    pip3 install -e ./clifford

If you are not running as ``sudo``, and your python installation is system-wide, you will need to pass ``--user`` to the invocations above.
