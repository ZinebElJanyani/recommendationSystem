import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#Load the dataset
data = pd.read_csv('dataset.csv')

#Load the user ratings
ratings = pd.read_csv('ratings.csv')

#Combine the product descriptions and categories into a single text field
data['text'] = data['description'] + ' ' + data['category']

#Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

#Vectorize the product text
text_vectors = vectorizer.fit_transform(data['text'])

#Compute the cosine similarity matrix
cosine_similarities = cosine_similarity(text_vectors)


# Define a function to get the top recommended products for a given user
def get_top_recommendations(user_id, k=15):
    # Get the user ratings for the given user
    user_ratings = ratings[ratings['user_id'] == user_id]

    # Filter out the products that the user has rated negatively
    user_ratingsPositive = user_ratings[user_ratings['rating'] >= 3]
    print(user_ratingsPositive)
    # Get the products with the highest ratings from the user
    top_products = user_ratingsPositive.sort_values(by='rating', ascending=False).head(4)['product_id'].tolist()
    print(top_products)
    top_products_indexs = [data['product_id'].to_list().index(y) for y in top_products ]

    # Get the cosine similarities between the top rated products and all other products
    top_similarities = cosine_similarities[:, [x - 1 for x in top_products_indexs]]

    # Compute the average similarity across all top rated products
    average_similarities = np.mean(top_similarities, axis=1)

    # Get the indices of the top k similar products
    top_indices = average_similarities.argsort()[::-1][:k]

    # Get the product IDs of the top k similar products
    top_product_ids = [data.iloc[i]['product_id'] for i in top_indices]

    # Exclude products that the user has already rated
    already_rated_products = user_ratings['product_id'].tolist()
    top_product_ids = [p for p in top_product_ids if p not in already_rated_products]

    # Return the top k recommended products
    return top_product_ids

