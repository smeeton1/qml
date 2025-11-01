FROM ubuntu:latest

WORKDIR /app

RUN apt-get update && apt-get install -y 
RUN apt-get install git nano python3 python3-pip -y 
RUN python3 -m pip config set global.break-system-packages true

RUN git clone https://github.com/smeeton1/qml.git
WORKDIR /app/qml
RUN pip3 install -r requirements.txt 

ENTRYPOINT ["/bin/bash"]
