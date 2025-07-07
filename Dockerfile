FROM continuumio/miniconda3
WORKDIR /home
COPY . /home
RUN apt update -y && apt upgrade -y && apt install -y nano gcc build-essential python3-dev
RUN pip install --upgrade pip setuptools wheel
RUN pip install -v -r requirements.txt
ENV PYTHONPATH=/home
CMD ["python", "-m", "pytest", "tests/"]
