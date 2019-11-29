FROM python:3.8-alpine

RUN pip install -r requirements.txt

COPY app/* .

CMD ["python", "./app.py"]
