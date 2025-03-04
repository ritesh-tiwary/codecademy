import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
reviews = pd.read_csv('reviews.csv')

# Explore the dataset
print(reviews.columns)
print(reviews.info())

# Transform 'recommended' column
print(reviews['recommended'].value_counts())
binary_dict = {True: 1, False: 0}
reviews['recommended'] = reviews['recommended'].map(binary_dict)
print(reviews['recommended'].value_counts())

# Transform 'rating' column
print(reviews['rating'].value_counts())
rating_dict = {
    'Loved it': 5,
    'Liked it': 4,
    'Was okay': 3,
    'Not great': 2,
    'Hated it': 1
}
reviews['rating'] = reviews['rating'].map(rating_dict)
print(reviews['rating'].value_counts())

# One-hot encode 'department_name'
print(reviews['department_name'].value_counts())
one_hot = pd.get_dummies(reviews['department_name'], prefix='dept')
reviews = reviews.join(one_hot)
print(reviews.columns)

# Transform 'review_date' to datetime
reviews['review_date'] = pd.to_datetime(reviews['review_date'])
print(reviews.dtypes['review_date'])

# Select numerical features
numerical_features = ['recommended', 'rating'] + list(one_hot.columns)
data = reviews[numerical_features]

# Reset index to 'clothing_id'
data.set_index(reviews['clothing_id'], inplace=True)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Convert back to DataFrame
scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
print(scaled_df.head())

