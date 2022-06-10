import functions
import tensorflow
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')

def plot_graphs(X, pred):


    fig = plt.figure(figsize=(8, 5))
    colors = {0: 'blue', 1: 'red'}
    for i in range(len(pred)):
        data = pd.DataFrame(X[i], columns=['Stage', 'Temp'])
        data.sort_values(by=['Stage'], inplace=True)
        sns.lineplot(data=data, x='Temp', y='Stage', color=colors[pred[i][0]], palette=['r', 'b'])
    fig.savefig(r'.\static\images\plot.png')
    plt.close()


def test(filepath, modelpath):
    df = functions.get_data_for_test(filepath)
    X, X_padded = functions.get_sequences(df)
    model = tensorflow.keras.models.load_model(modelpath)
    pred = model.predict(X_padded).round()
    predictions = get_predictions_for_excel(pred, X)
    df['Predictions'] = predictions
    if not os.path.exists('./predictions'):
        os.mkdir('./predictions')
    df.to_excel(f'./predictions/{os.path.basename(filepath.filename)}')
    plot_graphs(X, pred)

def get_predictions_for_excel(pred, X):
    predictions = []
    for x, y in zip(X, pred):
        prediction = 'good point' if y[0] == 0 else 'abnormality'
        predictions.extend(len(x) * [prediction])
    return predictions
