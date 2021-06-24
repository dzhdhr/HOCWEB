FROM python:3.8
WORKDIR /HOC
COPY requirements.txt ./
RUN pip3 install -r requirements.txt
ENV PYTHONUNBUFFERED=0 CUDA_VISIBLE_DEVICES=0
COPY . .
CMD ["gunicorn", "app:app", "-c", "./gunicorn.conf"]