FROM tensorflow/tensorflow:latest-gpu-py3

# Install all required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Add application files
COPY app/ /root/

EXPOSE 5000

WORKDIR /root/

ENTRYPOINT ["python", "-u"]
CMD ["./app.py"]
