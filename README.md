# Probabilistic Relational Agent-based Modeling (PRAM) Framework

A simulation framework that fuses relatinal probabilistic models and agent-based models.  This software is in the pre-alpha development stage, so feel free to play around with it, but keep in mind that it is constantly evolving and not all pieces may be working.


## Dependencies
- [Python 3.6](https://python.org)
- [attrs](https://github.com/python-attrs/attrs)
- [numpy](https://www.numpy.org)


## Setup
The following shell script creates a Python virtual environment (`venv`), activates it, downloads the source code of PRAM into it, and installs all dependencies.  When done, PRAM is executable in that `venv`.  Please note, that Python 3.6 is required.

```
#!/bin/sh

prj=pram

[[ -d $prj ]] && echo "Directory '$prj' already exists." && exit 1

python3 -m venv $prj
cd $prj
source ./bin/activate

git init
git remote add origin https://github.com/momacs/$prj
git pull origin master

[ -f requirements.txt ] && python -m pip install -r requirements.txt
```

The same result can be achieved in a fewer number of steps by using the [`momacs`](https://github.com/momacs/misc/momacs) command-line utility.  Specifically, the command of interest is `momacs pram make-venv`.  Another command (i.e., `momacs pram update-venv`) can be subsequently used to download the latest version of PRAM from the `venv`.

Once the PRAM `venv` has been activated, run the following from the `src` directory to see the results of a simple simulation:
```
python sim_01_simple.py
```


## License
This project is licensed under the [BSD License](LICENSE.md).
