#!/bin/bash

if [[ $- != *i* ]]
then
    echo "Please run in interactive mode, i.e. bash -i ...; aborting."
    exit 1
fi

# exit when any command fails
set -e

# create and activate conda env
conda create -p "/home/jleuschn/shared_envs/cinn_for_imaging_cuda102"
conda activate /home/jleuschn/shared_envs/cinn_for_imaging_cuda102

# install torch as suggested on https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

# install latest astra dev package
conda install astra-toolbox -c astra-toolbox/label/dev

# install pip packages
pip install pytorch-lightning https://github.com/odlgroup/odl/archive/master.zip https://github.com/ahendriksen/tomosipo/archive/develop.zip git+https://github.com/VLL-HD/FrEIA.git dival
