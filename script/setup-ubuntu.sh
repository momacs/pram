#!/bin/sh

do_venv=1

name=pram

[[ -d $name ]] && echo "Directory '$name' already exists." && exit 1

# Update the OS:
apt update
apt upgrade

# Install required packages:
apt install git gcc build-essential autoconf libtool pkg-config libgmp-dev libcairomm-1.0-dev libffi-dev libsparsehash-dev
apt install python3-dev python3.6-dev python3-numpy python3-pip python3-cairo python3-cairo-dev python3-cairocffi
apt install firefox firefox-geckodriver  # for Altair
apt install aptitude  # for GCAL
aptitude install libcgal-dev

# Install boost:
apt install libboost1.65-all-dev libboost1.65-dev

# Install graph-tool:
echo '# graph-tool:'                                                 | sudo tee -a /etc/apt/sources.list
echo 'deb http://downloads.skewed.de/apt/bionic binoic universe'     | sudo tee -a /etc/apt/sources.list
echo 'deb-src http://downloads.skewed.de/apt/bionic bionic universe' | sudo tee -a /etc/apt/sources.list

apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
apt update
apt install python-graph-tool python3-graph-tool

# Clone the repo, install Python dependencies, and run a simple PRAM simulation:
if [[ "$do_venv" == "1" ]]; then
	python3 -m venv $name
	cd $name
	source ./bin/activate
	
	git init
	git remote add origin https://github.com/momacs/$name
	git pull origin master
else
	git clone https://github.com/momacs/$name
	cd $name
done

pip3 install -r requirements.txt
cd src/sim/01-simple
python3 sim.py
