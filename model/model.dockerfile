FROM python:3.10.12-slim

WORKDIR /app

# Copiar archivos necesarios
COPY ./src /app/src
COPY ./artifacts /app/artifacts
COPY app.py /app/


COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y

RUN pip install -r requirements.txt

CMD ["python", "app.py"]