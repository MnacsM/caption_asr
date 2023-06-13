FROM python:3.9

WORKDIR /usr/src/app
ENV FLASK_APP=app

COPY /app/requirements.txt ./

# RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
RUN curl --proto '=https' --tlsv1.2 https://sh.rustup.rs > rustup.sh && sh rustup.sh -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
