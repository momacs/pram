#!/bin/bash

set -e

do_venv=1  # set to 0 to install without a venv
name=pram

if [ -d $name ]; then
    echo "Directory '$name' already exists."
    exit 1
fi

# Ask for intent confirmation:
if [ "$do_venv" -eq "1" ]; then
    read -p 'Install with Python venv? [y/n] ' res
    if [ "$res" != "y" -a "$res" != "Y" ]; then exit 0; fi
else
    read -p 'Install without Python venv (i.e., into system site directories)? [y/n] ' res
    if [ "$res" != "y" -a "$res" != "Y" ]; then exit 0; fi
fi

# Update the OS:
sudo apt -y update
sudo apt -y upgrade

# Install required packages:
sudo apt -y install git gcc build-essential autoconf libtool pkg-config libgmp-dev libcairomm-1.0-dev libffi-dev libsparsehash-dev
sudo apt -y install python3-dev python3.6-dev python3-pip python3-venv python3-numpy python3-cairo python3-cairo-dev python3-cairocffi
sudo apt -y install firefox firefox-geckodriver  # for Altair
sudo apt -y install aptitude  # for GCAL
sudo aptitude -y install libcgal-dev

# Install boost:
sudo apt -y install libboost1.65-all-dev libboost1.65-dev

# Install graph-tool:
echo '# graph-tool:'                                                 | sudo tee -a /etc/apt/sources.list
echo 'deb http://downloads.skewed.de/apt/bionic bionic universe'     | sudo tee -a /etc/apt/sources.list
echo 'deb-src http://downloads.skewed.de/apt/bionic bionic universe' | sudo tee -a /etc/apt/sources.list

sudo apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
sudo apt update
sudo apt -y install python-graph-tool python3-graph-tool

# Clone the repo, install Python dependencies, and run a simple PRAM simulation:
if [ "$do_venv" -eq "1" ]; then
    python3 -m venv $name
    cd $name
    source ./bin/activate
    
    git init
    git remote add origin https://github.com/momacs/$name
    git pull origin master
else
    git clone https://github.com/momacs/$name
    cd $name
fi

pip3 install -r requirements.txt
cd src/sim/01-simple
python3 sim.py
