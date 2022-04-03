#!/usr/bin/env python
from sklearn import datasets
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn2pmml import PMMLPipeline, sklearn2pmml

if __name__ == '__main__':
    # Read dataset
    data = datasets.load_iris(as_frame=True)
    features = data['data']
    target = data['target']

    # Features
    features.head()

    # Target
    target.head()

    pipeline = PMMLPipeline([('scaler', preprocessing.StandardScaler()), ('classifier', LogisticRegression())])
    pipeline.fit(features, target)
    pipeline.predict([[5.7, 2.8, 4.1, 1.3]])[0]

    # Export the model
    sklearn2pmml(pipeline, 'model.pmml', with_repr = True)
