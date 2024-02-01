FROM python:3.10-slim

WORKDIR /usr/src/app

RUN apt-get update && \
    apt-get install -y wget unzip && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt download_data.sh ./
COPY search_app ./search_app
COPY data/db ./data/db

RUN chmod +x ./download_data.sh && \
    ./download_data.sh

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "search_app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
