### Copyright 2003-2023. GenePattern Team @ Mesirov Lab - University of California, San Diego. All rights reserved.
# Disclaimer: the following Dockerfile is for future implementation, the current module uses the genepattern/notebook-python39:22.04 image.
FROM genepattern/notebook-python39:22.04

### Based off of ExampleModule Dockerfile found at https://github.com/genepattern/ExampleModule 
MAINTAINER Omar Halawa <ohalawa@ucsd.edu>

# -----------------------------------
# Creating a non-root user
RUN useradd -ms /bin/bash gpuser
USER gpuser
WORKDIR /home/gpuser

# Switching back to root to create dir
USER root
RUN mkdir /RandomForest \
    && chown gpuser /RandomForest

# Switching to non-root before exiting so that we don't run as root on the server, and copying all of the src files into the container.
USER gpuser
COPY src/*.py /RandomForest/

RUN /RandomForest/rnd_forest.py
# -----------------------------------

# docker build --rm https://github.com/omarhalawa3301/RandomForest.git#develop -f Dockerfile -t genepattern/randomforest:<tag>
# make sure this repo and tag match the manifest & don't forget to docker push!
# docker push genepattern/randomforest:<tag>

# you can use this command to run Docker and iterate locally (update for your paths and module name, of course)
# docker run --rm -it --user gpuser -v /c/Users/MyUSER/PathTo/RandomForest:/mnt/mydata:rw genepattern/randomforest:<tag> bash
