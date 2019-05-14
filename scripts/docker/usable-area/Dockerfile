FROM ingn/tensorflow-gpu-openslide:latest

RUN apt-get install -y vim

WORKDIR /
RUN ls
RUN cd svs_reader &&\
    pip3 install -e .

WORKDIR /
ADD deploy_usable_area.py /app/
ADD snapshots/* /app/snapshots/mobilenet_v2

RUN git clone https://github.com/nathanin/milk &&\
    cd milk &&\
    pip3 install -e .

WORKDIR /app

#ADD run_deploy.sh .
