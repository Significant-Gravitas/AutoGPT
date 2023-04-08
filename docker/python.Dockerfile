FROM python:3.11

WORKDIR /app
#COPY scripts/ /app
COPY docker-files/ /app/


COPY .env /app/.env
RUN . /app/.env


RUN pip install -r requirements.txt


