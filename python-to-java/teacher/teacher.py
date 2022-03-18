from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn2pmml import PMMLPipeline, sklearn2pmml

if __name__ == '__main__':
    # fetching data example
    data = load_diabetes(as_frame=True)
    features = data['data']
    target = data['target']

    # training the model
    pipeline = PMMLPipeline([('regressor', DecisionTreeRegressor())])
    pipeline.fit(features, target)

    # make a prediction
    pipeline.predict([[0.01809694, 0.00301924, 0.00511107, -0.00222774, -0.02633611, -0.02699205, 0.01550536,
                       -0.02104282, -0.02421066, -0.05492509]])[0]

    sklearn2pmml(pipeline, '/model.pmml', with_repr=True)
