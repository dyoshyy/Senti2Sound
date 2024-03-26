FROM python:3.10

# install python package
COPY pyproject.toml ./
RUN pip install poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

# mount dir
RUN mkdir -p /opt/mnt
WORKDIR /opt/mnt
COPY . /opt/mnt/

ENV FLASK_APP=src/app.py

# expose port
EXPOSE 8000

CMD ["gunicorn", "--bind" , ":8080", "--workers", "2", "src.app:app"]
