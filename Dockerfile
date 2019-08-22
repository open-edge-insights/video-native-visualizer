# Dockerfile for Visualizer
ARG EIS_VERSION
FROM ia_pybase:$EIS_VERSION as pybase
LABEL description="Visualizer image"

ARG HOST_TIME_ZONE=""

WORKDIR ${PY_WORK_DIR}

# Setting timezone inside the container
RUN echo "$HOST_TIME_ZONE" >/etc/timezone
RUN cat /etc/timezone
RUN apt-get update
RUN apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/${HOST_TIME_ZONE} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Installing dependencies
RUN apt-get install -y libmbedtls-dev \
                       build-essential \
                       libsm6 libxext6 libxrender-dev

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN apt-get install -y python3.6-tk

ARG EIS_USER_NAME
RUN adduser --quiet --disabled-password ${EIS_USER_NAME}

RUN touch visualize.log && \
    chown ${EIS_USER_NAME}:${EIS_USER_NAME} visualize.log &&  \
    chmod 777 visualize.log

RUN apt-get -y install git
ENV PY_ETCD3_VERSION cdc4c48bde88a795230a02aa574df84ed9ccfa52
RUN git clone https://github.com/kragniz/python-etcd3 && \
    cd python-etcd3 && \
    git checkout ${PY_ETCD3_VERSION} && \
    python3.6 setup.py install && \
    cd .. && \
    rm -rf python-etcd3

ENV PYTHONPATH ${PY_WORK_DIR}/

FROM ia_common:$EIS_VERSION as common

FROM pybase

COPY --from=common /libs ${PY_WORK_DIR}/libs
COPY --from=common /Util ${PY_WORK_DIR}/Util

RUN cd ./libs/EISMessageBus && \
    rm -rf build deps && \
    mkdir build && \
    cd build && \
    cmake -DWITH_PYTHON=ON .. && \
    make && \
    make install

COPY visualize.py .

ENTRYPOINT ["python3.6", "visualize.py"]
