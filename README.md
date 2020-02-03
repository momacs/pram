# Probabilistic Relational Agent-Based Modeling (PRAM) Framework

Probabilistic Relational Agent-based Models (PRAMs) is a modeling and simulation framework that puts agent-based models (ABMs) on a sound probabilistic foundation. When compared to equivalent ABMs, PRAMs are:

- More space- and time-efficient (due to the way it encodes agent population)
- More sound (due to being probabilistic)
- More expressive (due to being relational)

For more information see [documentation](https://momacs.github.io/pram/).

This software is in the pre-release stage.


## Dependencies - Core Library ([`src/pram`](src/pram))

- [Python 3.6](https://python.org)
- [xxhash](https://pypi.org/project/xxhash/)
- [dotmap](https://pypi.org/project/dotmap)
- [sortedcontainers](http://www.grantjenks.com/docs/sortedcontainers/index.html)
- [attrs](https://github.com/python-attrs/attrs)
- [dill](https://pypi.org/project/dill/)
- [numpy](https://www.numpy.org)
- [scipy](https://www.scipy.org)
- [matplotlib](https://matplotlib.org/)
- [graph-tool](https://graph-tool.skewed.de/)
- [pycairo](https://www.cairographics.org/pycairo/) (for `graph-tool`)
- [PyGObject](https://pygobject.readthedocs.io) (for `graph-tool`)
- [altair](https://altair-viz.github.io)
- [selenium](https://selenium-python.readthedocs.io/) (for saving `altair` graphs)
- [Gecko Driver](https://github.com/mozilla/geckodriver/releases) or [Chrome Driver](https://sites.google.com/a/chromium.org/chromedriver/) and a recent version of either of the respective Web browser (i.e., Firefox or Chrome; for saving `altair` graphs)
- [PyRQA](https://pypi.org/project/PyRQA/)
- [pybind11](https://pybind11.readthedocs.io/en/stable/) (for `PyRQA`)
- [PyOpenCL](https://documen.tician.de/pyopencl/) (for `PyRQA`)


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
### Ubuntu

The [`setup-ubuntu.sh`](https://github.com/momacs/pram/blob/master/script/setup-ubuntu.sh) script is the preferred method of deploying the package onto a fresh installation of the Ubuntu Server/Desktop (tested with 18.04 LTS).  The `do_env` variable controls whether the package and its dependencies are installed inside a Python `venv` (yes by default).  The script can be downloaded and run like so using either `wget`:

```
sh -c "$(wget https://raw.githubusercontent.com/momacs/pram/master/script/setup-ubuntu.sh -O -)"
```

or `curl`:

```
sh -c "$(curl -fsSL https://raw.githubusercontent.com/momacs/pram/master/script/setup-ubuntu.sh)"
```

### Python Package Only (`venv`)

The following shell script creates a Python virtual environment (`venv`), activates it, downloads the source code of PRAM into it, and installs all Python dependencies.  It does not however install the system-level dependencies (e.g., [`graph-tool`](https://graph-tool.skewed.de/)).

```
#!/bin/sh

name=pram

[[ -d $name ]] && echo "Directory '$name' already exists." && exit 1

python3 -m venv $name
cd $name
source ./bin/activate

git init
git remote add origin https://github.com/momacs/$name
git pull origin master

[ -f requirements.txt ] && python -m pip install -r requirements.txt
```

The same result can be achieved with one of the commands of the [`momacs`](https://github.com/momacs/misc) command-line utility.  Another command can be used subsequently to update an existing PRAM `venv` with the latest version of the source code from the present repository.  These two commands are, respectively:
```
momacs pram setup
momacs pram update
```

Once a PRAM `venv` has been activated, running the following will display results of a simple simulation:
```
python src/sim/01-simple/sim.py
```


## References

[Documentation](https://momacs.github.io/pram/)

Cohen, P.R. & Loboda, T.D. (2019) Probabilistic Relational Agent-Based Models.  _International Conference on Social Computing, Behavioral-Cultural Modeling & Prediction and Behavior Representation in Modeling and Simulation (BRiMS)_, Washington, DC, USA.  [PDF](https://github.com/momacs/pram/blob/master/pub/cohen-2019-brims.pdf)

Loboda, T.D. (2019) [Milestone 3 Report](https://github.com/momacs/pram/blob/master/pub/Milestone-3-Report.pdf).

Loboda, T.D. & Cohen, P.R. (2019) Probabilistic Relational Agent-Based Models.  Poster presented at the _International Conference on Social Computing, Behavioral-Cultural Modeling & Prediction and Behavior Representation in Modeling and Simulation (BRiMS)_, Washington, DC, USA.  [PDF](https://github.com/momacs/pram/blob/master/pub/loboda-2019-brims.pdf)


## License
This project is licensed under the [BSD License](LICENSE.md).
