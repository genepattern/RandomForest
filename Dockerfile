### Copyright 2003-2023. GenePattern Team @ Mesirov Lab - University of California, San Diego. All rights reserved.
#
# Work in progress.
# Currently, module uses genepattern/notebook-python39:22.04 image.
FROM python:3

# Based off of ExampleModule Dockerfile found at https://github.com/genepattern/ExampleModule 
LABEL maintainer="Omar Halawa ohalawa@ucsd.edu"

# Setting up proper environment, see ExampleModule Dockerfile for more info
# -----------------------------------
RUN useradd -ms /bin/bash gpuser
USER gpuser
WORKDIR /home/gpuser

USER root
RUN mkdir /RandomForest \
    && chown gpuser /RandomForest

USER gpuser
COPY src/*.py /RandomForest/
# -----------------------------------

# Ensuring up-to-date pip and importing necessary modules 
RUN pip install --upgrade pip && \
    pip install pandas && \
    pip install scikit-learn
    
# Running the program
RUN /RandomForest/rnd_forest.py
