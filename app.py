import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import streamlit as st
from matplotlib import cm
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
st.set_page_config(page_title="Visual Embeddings and Similarity", page_icon="🤖", layout="wide")

def main():
    # sidebar with openai api key and nomic token
    st.sidebar.title("Credentials")
    st.sidebar.write("OpenAI API Key")
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", value=os.getenv("OPENAI_API_KEY"))
    st.sidebar.write("Nomic Token")
    nomic_token = st.sidebar.text_input("Enter your Nomic Token", value=os.getenv("NOMIC_TOKEN"))

    openai.api_key = os.getenv("OPENAI_API_KEY")
    nomic.login(os.getenv("NOMIC_TOKEN"))

    # get data
    datafile_path = "food_review.csv"
    # show only columns ProductId, Score, Summary, Text, n_tokens, embedding
    df = pd.read_csv(datafile_path, usecols=[0,1,3, 5, 7, 8])
    st.title("Visual Embeddings and Similarity")
    st.write("Amazon food reviews dataset")
    st.write(df)
    
    st.write("Search similarity")
    form = st.form('Embeddings')
    question = form.text_input("Enter a sentence to search for semantic similarity", value="I love this soup")
    btn = form.form_submit_button("Run")

    if btn:
        # si openai api key no es none y nomic token no es none
        if openai_api_key is not None and nomic_token is not None:
            with st.spinner("Loading"):
                search_term_vector = get_embedding(question, engine="text-embedding-ada-002")
                search_term_vector = np.array(search_term_vector)

                matrix = np.array(df.embedding.apply(literal_eval).to_list())

                # Compute distances to the search_term_vector
                distances = np.linalg.norm(matrix - search_term_vector, axis=1)
                df['distance_to_search_term'] = distances

                # Normalize the distances to range 0-1 for coloring
                df['normalized_distance'] = (df['distance_to_search_term'] - df['distance_to_search_term'].min()) / (df['distance_to_search_term'].max() - df['distance_to_search_term'].min())

                # 2D visualization
                # Create a t-SNE model and transform the data
                # tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='random', learning_rate=200)
                # vis_dims = tsne.fit_transform(matrix)

                # colors = cm.rainbow(df['normalized_distance'])
                # x = [x for x,y in vis_dims]
                # y = [y for x,y in vis_dims]

                # # Plot points with colors corresponding to their distance from search_term_vector
                # plt2.scatter(x, y, color=colors, alpha=0.3)

                # # Set title and plot
                # plt2.title("Similarity to search term visualized in language using t-SNE")
                
                #3D visualization PCA
                pca = PCA(n_components=3)
                vis_dims_pca = pca.fit_transform(matrix)
                question_vis = vis_dims_pca.tolist()

                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(projection='3d')
                cmap = plt.get_cmap("tab20")

                # Plot question_vis
                ax.scatter(question_vis[0][0], question_vis[0][1], question_vis[0][2], color=cmap(0), s=100, label="Search term")
                # Plot other points
                for i, point in enumerate(vis_dims_pca):
                    ax.scatter(point[0], point[1], point[2], color=cmap(df['normalized_distance'][i]), alpha=0.3)
                ax.set_title("Similarity to search term visualized in language using PCA")
                ax.legend()
                plt.show()
               
                
                # Convert 'embedding' column to numpy arrays
                df['embedding'] = df['embedding'].apply(lambda x: np.array(literal_eval(x)))
                df["similarities"] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))
                
                st.title("Visual embedding of the search term and the 20 most similar sentences")
                #create two columns
                col1, col2 = st.columns(2)
                #col1
                #show st.plot in col1
                col1.pyplot(plt)

                #col2
                # #show st.plot in col2
                # col2.pyplot(fig)
                
                #col3
                #show df in col2, but only the columns, text and similarities
                col2.write(df[['similarities','Text']].sort_values("similarities", ascending=False).head(20))
                
                # Convert to a list of lists of floats
                st.title("Nomic mappping embeddings")
                embeddings = np.array(df.embedding.to_list())            
                df = df.drop('embedding', axis=1)
                df = df.rename(columns={'Unnamed: 0': 'id'})

                data = df.to_dict('records')
                project = atlas.map_embeddings(embeddings=embeddings, data=data,
                                            id_field='id',
                                            colorable_fields=['Score'])
                # Convert project to a string before getting link information
                project_str = str(project)

                st.text(project_str)
                # Split the project string at the colon and take the second part (index 1)
                project_link = project_str.split(':', 1)[1]

                # Trim any leading or trailing whitespace
                project_link = project_link.strip()

                # Crea un iframe con la URL y muéstralo con Streamlit
                st.markdown(f'<iframe src="{project_link}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)
        else:
            st.write("Please enter your OpenAI API Key and Nomic Token in the sidebar")
if __name__ == "__main__":
    main()