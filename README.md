# LangChain Agent AI
This Python script uses the OpenAI API to analyze Amazon food reviews by encoding them into embeddings and conducting semantic similarity searches.

## Code Overview

The code begins by importing necessary libraries and loading environmental variables that store sensitive information. It then defines a `main()` function, which is later invoked to run the program.

In `main()`, we first read a CSV file containing Amazon food reviews into a pandas dataframe. We then set the webpage title and icon using Streamlit, another necessary library for this script.

Using Streamlit forms, the code collects a search sentence and, upon form submission, applies an embedding function from OpenAI to the sentence to create a vector representation of its semantics - a semantic search term.

The high-dimensional embeddings are then visualized in 2D utilizing the t-SNE algorithm, with different Amazon ratings being displayed in different colors.

Upon calculation of cosine similarities between the search term vector and individual review embeddings, the 20 reviews with the highest similarities are displayed, showcasing the power of semantic similarity search.

Lastly, the script maps embeddings to the `nomic` database, allowing the program to interact with the `nomic` embedding mapping and storage tool.

## Requirements

- Python 3
- pandas
- NumPy
- scikit-learn
- Streamlit
- dotenv
- Nomic
- Openai

## How to Use
1. Install all dependencies `pip install -r requirements.txt`
2. Set up environment variables for the OPENAI_API_KEY & NUMIC_TOKEN for authentication
3. Run `streamlit run app.py`
4. Open a web browser to the provided localhost URL
5. Interact with the visualizations and use the semantic search functionality on the webpage
