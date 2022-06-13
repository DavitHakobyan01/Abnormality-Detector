import functions
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('agg')


def plot_graphs(X, pred, date):
    fig = plt.figure(figsize=(8, 5))
    colors = {0: 'blue', 1: 'red'}
    # for i in range(len(pred)):
    #     data = pd.DataFrame(X[i], columns=['Stage', 'Temp'])
    #     data.sort_values(by=['Stage'], inplace=True)
    X.sort_values(by=['Well Temperature Survey Profile Measured Depth'], inplace=True)
    sns.lineplot(data=X,
                 x='Well Temperature Survey Profile Measured Depth',
                 y='Well Temperature Survey Temperature',
                 marker='o',
                 color=colors[pred])
    plt.grid()
    fig.savefig(r'.\static\images\plot.png')
    plt.close()


def predict(filepath, modelpath):
    date, df = functions.get_data_for_prediction(filepath)
    X, X_padded = functions.get_sequences(df)
    model = tensorflow.keras.models.load_model(modelpath)
    pred = model.predict(X_padded)
    plot_graphs(X, int(pred.round().item()), date)
    return date, pred.round(decimals=3).item()

