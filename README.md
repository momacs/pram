# Probabilistic Relational Agent-Based Models (PRAMs)

Probabilistic Relational Agent-based Models (PRAMs) is a modeling and simulation framework that puts agent-based models (ABMs) on a sound probabilistic foundation. When compared to equivalent ABMs, PRAMs are:

- More space- and time-efficient (due to the way it encodes agent population)
- More sound (due to being probabilistic)
- More expressive (due to being relational)

For more information see [documentation](https://momacs.github.io/pram/).

This software is in the pre-release stage.


## Dependencies - Core Library ([`src/pram`](src/pram))

- [Python 3](https://python.org)
- [altair](https://altair-viz.github.io)
- [altair-saver](https://pypi.org/project/altair-saver)
- [attrs](https://github.com/python-attrs/attrs)
- [cloudpickle](https://github.com/cloudpipe/cloudpickle)
- [dotmap](https://pypi.org/project/dotmap)
- [iteround](https://pypi.org/project/iteround)
- [matplotlib](https://matplotlib.org)
- [numpy](https://www.numpy.org)
- [psutil](https://github.com/giampaolo/psutil)
- [psycopg2](https://www.psycopg.org/)
- [pybind11](https://pybind11.readthedocs.io/en/stable) (for `PyRQA`)
- [PyOpenCL](https://documen.tician.de/pyopencl) (for `PyRQA`)
- [PyRQA](https://pypi.org/project/PyRQA)
- [ray](https://docs.ray.io)
- [scipy](https://www.scipy.org)
- [sortedcontainers](http://www.grantjenks.com/docs/sortedcontainers/index.html)
- [tqdm](https://tqdm.github.io)
- [xxhash](https://pypi.org/project/xxhash)
- [selenium](https://selenium-python.readthedocs.io) (for saving `altair` graphs)
- [Gecko Driver](https://github.com/mozilla/geckodriver/releases) or [Chrome Driver](https://sites.google.com/a/chromium.org/chromedriver) and a recent version of either of the respective Web browser (i.e., Firefox or Chrome; for saving `altair` graphs)


## Dependencies - Simulation Library ([`src/sim`](src/sim))

None


## Dependencies - The Web App ([`src/web`](src/web))

Backend:
- [Flask](http://flask.pocoo.org)
- [Celery](http://www.celeryproject.org)
- [Redis](https://redis.io)
- [PSUtil](https://github.com/giampaolo/psutil)

Front-end:
- [Materialize](https://materializecss.com)
- [jQuery](https://jquery.com)
- [Fuse.js](https://fusejs.io)


## Setup

You can install PyPRAM like so:

```
pip install git+https://github.com/momacs/pram.git
```

To install all extra dependencies instead, do:

```
pip install git+https://github.com/momacs/pram.git#egg=pram[all]
```

Remember to activate your `venv` of choice unless you want to go system-level.

### The `momacs` Utility

To create a new `venv` and install PyPRAM inside of it, use the [`momacs`](https://github.com/momacs/misc) command-line utility like so:

```
momacs app-pram setup
```

### Ubuntu

The [`setup-ubuntu.sh`](https://github.com/momacs/pram/blob/master/script/setup-ubuntu.sh) script is the preferred method of deploying the package and all its dependencies (including the system-level ones) onto a fresh installation of the Ubuntu Server/Desktop (tested with 18.04 LTS).  This is ideal for initializing virtual machine images.  The `do_env` variable controls whether the package and its dependencies are installed inside a Python `venv` (yes by default).  The setup script can be downloaded and executed like so:

```
sh -c "$(wget https://raw.githubusercontent.com/momacs/pram/master/script/setup-ubuntu.sh -O -)"
```


## References

[Documentation](https://momacs.github.io/pram/)

Cohen, P.R. & Loboda, T.D. (2019) Probabilistic Relational Agent-Based Models.  _International Conference on Social Computing, Behavioral-Cultural Modeling & Prediction and Behavior Representation in Modeling and Simulation (BRiMS)_, Washington, DC, USA.  [PDF](https://github.com/momacs/pram/blob/master/pub/cohen-2019-brims.pdf)

Loboda, T.D. (2019) [Milestone 3 Report](https://github.com/momacs/pram/blob/master/pub/Milestone-3-Report.pdf).

Loboda, T.D. & Cohen, P.R. (2019) Probabilistic Relational Agent-Based Models.  Poster presented at the _International Conference on Social Computing, Behavioral-Cultural Modeling & Prediction and Behavior Representation in Modeling and Simulation (BRiMS)_, Washington, DC, USA.  [PDF](https://github.com/momacs/pram/blob/master/pub/loboda-2019-brims.pdf)


## License
This project is licensed under the [BSD License](LICENSE.md).
