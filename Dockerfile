FROM continuumio/miniconda3
WORKDIR /home
COPY . /home
RUN apt update -y && apt upgrade -y && apt install -y nano gcc build-essential python3-dev -qq
RUN pip install --upgrade pip setuptools wheel -q
RUN pip install -v -r requirements.txt -q
ENV PYTHONPATH=/home
CMD ["python", "-m", "pytest", "tests/"]
