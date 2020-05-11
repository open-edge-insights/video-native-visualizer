# Dockerfile for Visualizer
ARG EIS_VERSION
FROM ia_eisbase:$EIS_VERSION as eisbase
LABEL description="Visualizer image"

WORKDIR ${PY_WORK_DIR}

# Installing dependencies
RUN apt-get install -y libgl1-mesa-dev python3-pyqt5

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ARG EIS_USER_NAME
RUN adduser --quiet --disabled-password ${EIS_USER_NAME}

ENV PYTHONPATH ${PY_WORK_DIR}/

FROM ia_common:$EIS_VERSION as common

FROM eisbase

COPY --from=common ${GO_WORK_DIR}/common/libs ${PY_WORK_DIR}/libs
COPY --from=common ${GO_WORK_DIR}/common/util ${PY_WORK_DIR}/util
COPY --from=common ${GO_WORK_DIR}/common/cmake ${PY_WORK_DIR}/common/cmake
COPY --from=common /usr/local/lib /usr/local/lib
COPY --from=common /usr/local/lib/python3.6/dist-packages/ /usr/local/lib/python3.6/dist-packages

COPY . .

#Removing build dependencies
RUN apt-get remove -y wget && \
    apt-get remove -y git && \
    apt-get remove curl && \
    apt-get autoremove -y

ENTRYPOINT ["python3.6", "visualize.py"]
