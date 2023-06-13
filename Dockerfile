FROM python:3.10.4

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY ./app/requirements.txt ./

# rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# WORKDIR /usr/src/app
# ENV FLASK_APP=app

# COPY /app/requirements.txt ./

# # RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# RUN curl --proto '=https' --tlsv1.2 https://sh.rustup.rs > rustup.sh && sh rustup.sh -y

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
