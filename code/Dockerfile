FROM python:3.9

WORKDIR .
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY augmentations/ augmentations/
COPY experiments/ experiments/
COPY pipelines/ pipelines/
COPY utils/ utils/
# Сюда вставить нужный лаунчер
COPY common_launcher.py ./launcher.py

# command to run on container start
CMD ["python", "./launcher.py", "data/config.json"]
