import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

#Import models from scikit learn module:
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler #not currently used
from scipy.stats import zscore #not needed but built in
#https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
df = pd.read_csv('breast_cancer_data.csv')
#encode malignant as 1, benign as 0
df['diagnosis'] = df['diagnosis'].replace({'M':1,'B':0})

print(df.head())

#imports/load data
predictor_var = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
df.head()

#1.Print distinct diagnosis values and frequency in dataset
print(df.diagnosis.value_counts())

#2. Test if the number of unique IDs is equal to sample size, i.e. no repeated patients
unique_ids = df.id.nunique()==df.id.count()
print(unique_ids)

#3. At a maximum, there should be no more than the smallest class size divided by 10 number of features.
max_features = min(df.diagnosis.value_counts()/10)
print(max_features)


#4. Uncomment the code to see which features have extreme outliers:
sns.boxplot(data=np.log(df[predictor_var]+.01).apply(zscore))
plt.xticks(rotation=45);
plt.show()
plt.close()

#5. Uncomment the code to remove the samples with extreme fractal_dimensions_mean values:
q_hi  = df["fractal_dimension_mean"].quantile(0.99)
df_filtered = df[(df["fractal_dimension_mean"] < q_hi)]

#6. Run the boxplot again but with the filtered dataframe:
sns.boxplot(data=np.log(df_filtered[predictor_var]+.01).apply(zscore))
plt.xticks(rotation=45);
plt.show()
plt.close()
