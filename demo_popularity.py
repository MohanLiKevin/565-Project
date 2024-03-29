import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DELTA = 1
ALPHA = 3
BETA = 3
GAMMA = 0.5


def popularity_calculation(raw_data):
    
    tmp1 = DELTA * np.log10(raw_data[:,0] + 1)
    tmp2 = ALPHA * raw_data[:,1] / (raw_data[:,0] + 1)
    tmp3 = BETA * raw_data[:,2] / (raw_data[:,0] + 1)
    tmp4 = GAMMA * raw_data[:,3] / (raw_data[:,0] + 1)
    
    numerator = 10 * (tmp1 + tmp2 - tmp3 + tmp4)
    denominator = DELTA + ALPHA + BETA + GAMMA
    
    result =  numerator / denominator
    
    return result.reshape(-1, 1)

csv_file_path = 'US_youtube_trending_data.csv'
data = pd.read_csv(csv_file_path, usecols=['tags','view_count', 'likes', 'dislikes', 'comment_count'])
df = pd.read_csv(csv_file_path, usecols=['view_count', 'likes', 'dislikes', 'comment_count'])

raw_data = df.values
print(raw_data)
popularity_scores = popularity_calculation(raw_data)

data['Popularity'] = popularity_scores

print(data.head())

output_file_path = 'Popularity.csv'
data.to_csv(output_file_path, index=False)   
    
    
    
    
    
    
    
    # popularity_output = np.array([[30],
    #                             [1]])

    # print(popularity_input.shape)
    # print(popularity_output.shape)



# # Sample label
# label = "This is a sample label for TF-IDF vectorization"
# label = ['apple', 'banana', 'cat', 'dog', 'desk', 'chair']
# # Create a TF-IDF vectorizer
# vectorizer = TfidfVectorizer()

# # Fit the vectorizer on the label (you can also pass a list of labels if you have multiple)
# vectorizer.fit(label)

# # Transform the label into a TF-IDF vector
# vector = vectorizer.transform(label)

# vector = vector.toarray()
# # Print the TF-IDF vector
# print(vector.shape)

# kmeans = KMeans(n_clusters=3, random_state=0).fit(vector)
# print(kmeans.labels_)

