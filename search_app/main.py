import streamlit as st
import open_clip
from qdrant_client import QdrantClient
import torch


st.set_page_config(
    page_title="search system",
    page_icon="👋"
    )
st.write("# Welcome to text2image search system! 👋")

COLLECTION = "adv"
DB_PATH = "data/db"
MODEL_NAME = "ViT-B-16-SigLIP"
WEIGHTS = "webli"

top_k = st.sidebar.number_input('top_k', min_value=1, max_value=20, value=5)


@st.cache_resource()
def get_model_and_tokenizer():
    model = open_clip.create_model(
        model_name=MODEL_NAME,
        pretrained=WEIGHTS,
    )
    model = model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, tokenizer


@st.cache_resource()
def get_client():
    return QdrantClient(path=DB_PATH)


model, tokenizer = get_model_and_tokenizer()
client = get_client()


@st.cache_data()
def get_images(query, top_k):
    with torch.no_grad(), torch.cuda.amp.autocast():
        text = tokenizer(query)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    response = client.search(
        collection_name=COLLECTION,
        query_vector=text_features.tolist()[0],
        limit=top_k,
    )
    images = [image.payload['path'] for image in response]
    return images


query = st.text_input("Enter Query", placeholder="query")
if st.button('Get answer') and query:
    images = get_images(query, top_k)
    for image in images:
        st.image(image, width=400)
