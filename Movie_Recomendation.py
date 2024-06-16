import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Create a synthetic dataset
data = {
    'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4],
    'rating': [4, 5, 3, 4, 2, 5, 5, 4, 3]
}

df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Load data into Surprise Dataset
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'movie_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Build and train the SVD model
model = SVD()
model.fit(trainset)

# Perform cross-validation
cross_validate(model, data, measures=['RMSE'], cv=5, verbose=True)

# Predict ratings for the test set
predictions = model.test(testset)

# Calculate RMSE
mse = mean_squared_error([pred.r_ui for pred in predictions], [pred.est for pred in predictions])
rmse = np.sqrt(mse)
print('Root Mean Squared Error:', rmse)

# Function to get movie recommendations for a user
def get_movie_recommendations(model, user_id, movie_ids, num_recommendations=5):
    # Predict ratings for all movies for the given user
    predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids]
    
    # Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get top-N movie IDs
    top_movie_ids = [pred.iid for pred in predictions[:num_recommendations]]
    
    return top_movie_ids

# Example usage
all_movie_ids = df['movie_id'].unique()
recommended_movies = get_movie_recommendations(model, user_id=1, movie_ids=all_movie_ids)
print('Recommended Movies:', recommended_movies)
