FROM python:3.10

WORKDIR /app

RUN pip install --upgrade pip

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8080/tcp

CMD ["python", "main.py"]