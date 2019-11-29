FROM python:3.8-alpine

# Install all required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add application files
COPY app/* .

CMD ["python", "./app.py"]
