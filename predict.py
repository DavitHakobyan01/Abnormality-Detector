import functions
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend('agg')


def plot_graphs(X, prediction):
    fig = plt.figure(figsize=(8, 5))
    color = 'blue' if prediction == 0 else 'red'
    X.sort_values(by=['Well Temperature Survey Profile Measured Depth'], inplace=True)
    sns.lineplot(data=X, x='Well Temperature Survey Profile Measured Depth',
                 y='Well Temperature Survey Temperature', marker='o', color=color)
    plt.grid()
    fig.savefig(r'.\static\images\plot.png')
    plt.close()


def predict(filepath: object, model_path: str):
    date, df = functions.get_data_for_prediction(filepath)
    X, X_padded = functions.get_sequences(df)

    model = tensorflow.keras.models.load_model(model_path)
    pred = model.predict(X_padded)

    plot_graphs(X, int(pred.round().item()), date)

    return date, pred.round(decimals=3).item()

