FROM python:3.10.13
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE $PORT    
CMD gunicorn --works=4 --bind 0.0.0.0:$PORT app:app

