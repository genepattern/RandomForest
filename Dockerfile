### Copyright 2003-2023. GenePattern Team @ Mesirov Lab - University of California, San Diego. All rights reserved.
FROM python:3

### Based off of ExampleModule Dockerfile found at https://github.com/genepattern/ExampleModule 
MAINTAINER Omar Halawa <ohalawa@ucsd.edu>

### Ensuring up-to-date pip and importing necessary modules 
RUN pip install --upgrade pip && \
    pip install pandas
    pip install scikit-learn
