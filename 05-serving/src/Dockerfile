FROM python:3.9.16-slim

RUN pip install pipenv==2023.3.18

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "./"]
COPY ["/models/", "./models/"]

# Note: Make sure to run container with -p 9696:9696 to properly use this
EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]