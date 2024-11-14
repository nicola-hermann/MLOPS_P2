FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip 
RUN pip install .

CMD ["python", "run.py"]