FROM python:3.8.5-slim-buster
WORKDIR /app
COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY ./nltk_data /usr/local/nltk_data
EXPOSE 8080
CMD ["python3","main.py"]