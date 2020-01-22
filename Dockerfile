# Dockerfile for Visualizer
ARG EIS_VERSION
FROM ia_eisbase:$EIS_VERSION as eisbase
LABEL description="Visualizer image"

ARG HOST_TIME_ZONE=""

WORKDIR ${PY_WORK_DIR}

ARG DEBIAN_FRONTEND=noninteractive
# Setting timezone inside the container
RUN echo "$HOST_TIME_ZONE" >/etc/timezone
RUN cat /etc/timezone
RUN apt-get update && \
    apt-get install -y tzdata && \
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

#Removing build dependencies
RUN apt-get remove -y wget && \
    apt-get remove -y git && \
    apt-get remove curl && \
    apt-get autoremove -y

FROM ia_common:$EIS_VERSION as common

FROM eisbase

COPY --from=common ${GO_WORK_DIR}/common/libs ${PY_WORK_DIR}/libs
COPY --from=common ${GO_WORK_DIR}/common/util ${PY_WORK_DIR}/util
COPY --from=common ${GO_WORK_DIR}/common/cmake ${PY_WORK_DIR}/common/cmake
COPY --from=common /usr/local/lib /usr/local/lib
COPY --from=common /usr/local/lib/python3.6/dist-packages/ /usr/local/lib/python3.6/dist-packages

COPY . .

ENTRYPOINT ["python3.6", "visualize.py"]
