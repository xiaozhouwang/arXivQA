import streamlit as st
from utils import Embedding, get_data_embed, get_query_matches

st.header("arXiv Paper Search:")
paper_list, paper_embed = get_data_embed()

query = st.text_input(label="Search Query Input:", value="attention in RNN")
topK = st.number_input(label="Number of Matches:", min_value=1, max_value=100, value=5)
result = get_query_matches(query, paper_list, paper_embed, topK)
for i, r in enumerate(result):
    st.subheader(f'Paper {i+1}:')
    st.caption("Title:")
    st.write(r.split("abstract:")[0].replace("title:", ""))
    st.caption("Abstract:")
    st.write(r.split("abstract:")[1])

