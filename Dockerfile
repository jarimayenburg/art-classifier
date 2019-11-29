FROM tensorflow/tensorflow:latest-gpu-py3

# Install all required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add application files
COPY app/* .

CMD ["python", "-u", "./app.py"]
