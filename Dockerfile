# Copyright (c) 2020 Intel Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Dockerfile for Visualizer

ARG EII_VERSION
ARG DOCKER_REGISTRY
FROM ${DOCKER_REGISTRY}ia_eiibase:$EII_VERSION as eiibase
LABEL description="Visualizer image"

WORKDIR ${PY_WORK_DIR}

# Installing dependencies
RUN apt-get install -y python3.6-tk

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ARG EII_USER_NAME
RUN adduser --quiet --disabled-password ${EII_USER_NAME}

ENV PYTHONPATH ${PY_WORK_DIR}/

FROM ${DOCKER_REGISTRY}ia_common:$EII_VERSION as common

FROM eiibase

COPY --from=common ${GO_WORK_DIR}/common/libs ${PY_WORK_DIR}/libs
COPY --from=common ${GO_WORK_DIR}/common/util ${PY_WORK_DIR}/util
COPY --from=common ${GO_WORK_DIR}/common/cmake ${PY_WORK_DIR}/common/cmake
COPY --from=common /usr/local/lib /usr/local/lib
COPY --from=common /usr/local/lib/python3.6/dist-packages/ /usr/local/lib/python3.6/dist-packages

COPY . .

ARG EII_UID
RUN mkdir -p ${EII_INSTALL_PATH}/saved_images && \
    chown -R ${EII_UID}:${EII_UID} ${EII_INSTALL_PATH}/saved_images && \
    chmod 760 ${EII_INSTALL_PATH}/saved_images

#Removing build dependencies
RUN apt-get remove -y wget && \
    apt-get remove -y git && \
    apt-get remove curl && \
    apt-get autoremove -y

HEALTHCHECK NONE

ENTRYPOINT ["python3.6", "visualize.py"]
