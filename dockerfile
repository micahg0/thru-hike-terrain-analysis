# Ubuntu 18.04 (bionic)
#ARG BASE_CONTAINER=jupyter/scipy-notebook
FROM jupyter/scipy-notebook:94fdd01b492f

USER root

COPY environment.yml ./tmp/environment.yml

RUN conda env update -n base --file ./tmp/environment.yml

RUN fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

USER $NB_UID
