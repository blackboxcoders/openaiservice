FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 8003

CMD [ "python", "langchain_service.py"] 