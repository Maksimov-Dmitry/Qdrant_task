Text2Image System
==============================

This project aims to develop a system capable of searching for similar images based on textual queries. This repository contains the Streamlit application for the system and the code for model evaluation.

## Running the Text2Image Application
I highly recommend running the app in a Docker container. To do so, execute the following commands in the root directory of the project:
~~~
docker build -t search_app .
docker run -p 8501:8501 search_app
~~~
Afterward, open the app in your browser by typing `localhost:8501` in the address bar.

## Running the Evaluation
Before creating the Streamlit app, I had to create a pipeline to evaluate the model. The pipeline consists of the following steps:
1. Download the unlabeled dataset
2. Create labels for the dataset
3. Check the model's performance on the labeled dataset
4. Create a vector database if the model's performance is satisfactory

To run the above pipeline, execute the following commands in the root directory of the project:

1. Prepare environment
I have used python 3.10.11 for this project.
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
2. Download the dataset
~~~
bash download_dataset.sh
~~~
3. Create labels for the dataset
There are two ways to create labels for the dataset:
    - Using ImageNet models:
        ~~~
        python -m src.make_dataset dataset_model=imagenet output_file=data/processed/dataset_openai.jsonl
        ~~~
    - Using OpenAI models:
        Before running the command below, you need to create `.env` file in the root directory of the project and add OPENAI_API_KEY variable to it. Since each request to the API costs money, you can specify the number of requests you want to make using subset variable
        ~~~
        python -m src.make_dataset dataset_model=openai subset=100 output_file=data/processed/dataset_openai.jsonl
        ~~~
4. Check the model's performance on the labeled dataset
Specify the path to the dataset created in step 3 and the path to the output file.
~~~
python -m src.evaluation_pipeline dataset=data/processed/dataset_openai.jsonl output_file=results/eval_openai.jsonl
~~~
5. Create vector DB
~~~
python -m src.create_vector_db
~~~

P.S. In steps 3 and 4, you can specify the device for running the model using the 'device' variable. The default value is 'mps'. To change it, specify the device in the command line using Hydra notation, e.g., `device=cpu`.

## Report
For a comprehensive understanding of the methodologies, experiments, and results related to this project, refer to the detailed report available at:
[REPORT.md](reports/REPORT.md)
