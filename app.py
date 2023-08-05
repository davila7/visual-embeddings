import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from sklearn.manifold import TSNE
import streamlit as st
import pandas as pd
import numpy as np
from ast import literal_eval
import nomic
from nomic import atlas
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from dotenv import load_dotenv
load_dotenv()

MODEL = "text-embedding-ada-002"
openai.api_key = os.getenv("OPENAI_API_KEY")
nomic.login(os.getenv("NUMIC_TOKEN"))

def main():
    # get data
    datafile_path = "food_review.csv"
    df = pd.read_csv(datafile_path)

    st.set_page_config(page_title="Langchain Agent AI", page_icon="🤖", layout="wide")
    st.title("Show Embeddings and Similarity")
    st.write("Amazon food reviews dataset")
    st.write(df)
    
    st.write("Search similarity")
    form = st.form('Embeddings')
    question = form.text_input("Enter a sentence to search for semantic similarity", value="I love this soup")
    btn = form.form_submit_button("Run")

    if btn:
        with st.spinner("Loading"):
            # df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
            # df.to_csv('word_embeddings.csv')

            search_term_vector = get_embedding(question, engine="text-embedding-ada-002")
            search_term_vector = np.array(search_term_vector)

            # 2D visualization
            # Convert to a list of lists of floats
            matrix = np.array(df.embedding.apply(literal_eval).to_list())

            # Create a t-SNE model and transform the data
            tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
            vis_dims = tsne.fit_transform(matrix)
            vis_dims.shape

            colors = ["red", "darkorange", "gold", "turquoise", "darkgreen"]
            x = [x for x,y in vis_dims]
            y = [y for x,y in vis_dims]
            color_indices = df.Score.values - 1

            # Create colormap
            colormap = matplotlib.colors.ListedColormap(colors)
            plt.scatter(x, y, c=color_indices, cmap=colormap, alpha=0.3)
            
            # Loop over 0-4 scores
            for score in [0,1,2,3,4]:
                avg_x = np.array(x)[df.Score-1==score].mean()
                avg_y = np.array(y)[df.Score-1==score].mean()
                color = colors[score]
                plt.scatter(avg_x, avg_y, marker='x', color=color, s=100)
            
            # Set title and plot
            plt.title("Amazon ratings visualized in language using t-SNE")
            st.pyplot(plt)
            
            # Convert 'embedding' column to numpy arrays
            df['embedding'] = df['embedding'].apply(lambda x: np.array(literal_eval(x)))
            df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
            
            st.write("Embedding + similarity")
            #create two columns
            col1, col2 = st.columns(2)
            #first column
            col1.write("Embedding")
            col1.write(search_term_vector)
            #second column
            col2.write("Similarity")
            col2.write(df.sort_values("similarities", ascending=False).head(20))
            
            # Convert to a list of lists of floats
            embeddings = np.array(df.embedding.to_list())            
            df = df.drop('embedding', axis=1)
            df = df.rename(columns={'Unnamed: 0': 'id'})

            data = df.to_dict('records')
            project = atlas.map_embeddings(embeddings=embeddings, data=data,
                                        id_field='id',
                                        colorable_fields=['Score'])
            st.write(project)

if __name__ == "__main__":
    main()