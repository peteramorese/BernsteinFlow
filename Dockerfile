FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /root/BernsteinFlow

#RUN apt-get -y update && apt -y install git

# Copy requirements and install dependencies
COPY requirements.txt .

# Install apt dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

