FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install mlflow
COPY . /app
EXPOSE 5001
ENTRYPOINT ["uvicorn", "Serve:app","--host", "0.0.0.0" , "--port", "5001", "--reload"]