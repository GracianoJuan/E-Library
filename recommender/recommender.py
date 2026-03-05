import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBasedRecommender:
    def __init__(self, data_path='../dataset/amazon_books.csv'):
        self.data_path = data_path
        self.df = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.indices = None
        self.load_data()
        self.preprocess_data()
        self.build_tfidf()

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        # Clean data similar to exploration
        self.df = self.df.drop_duplicates()
        self.df['rating'] = self.df['rating'].str.extract(r'(\d+\.\d+)').astype(float)
        self.df['description'] = self.df['description'].fillna('')
        self.df['content'] = self.df['title'] + ' ' + self.df['description'] + ' ' + self.df['author']

    def preprocess_data(self):
        # Create indices mapping
        self.indices = pd.Series(self.df.index, index=self.df['title']).drop_duplicates()

    def build_tfidf(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['content'])

    def get_recommendations(self, title, num_recommendations=5):
        if title not in self.indices:
            return f"Book '{title}' not found in the dataset."

        idx = self.indices[title]
        sim_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations+1]  # Exclude itself

        book_indices = [i[0] for i in sim_scores]
        recommendations = self.df.iloc[book_indices][['title', 'author', 'category', 'rating']]
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        return recommendations
