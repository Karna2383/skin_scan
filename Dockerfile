FROM python:3.12.11-slim
COPY skin_scan /skin_scan
COPY requirements.txt /requirements.txt
COPY preprocessing_pipeline/preprocessor_joblib preprocessing_pipeline/preprocessor_joblib
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn skin_scan.fast:app --host 0.0.0.0 --port $PORT
