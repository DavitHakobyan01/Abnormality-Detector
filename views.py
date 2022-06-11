from flask import Blueprint, render_template, request
import os
import predict

views = Blueprint(__name__, 'views')
model_list = os.listdir(r'.\models')

@views.route("/")
def home():
    return render_template("index.html", models=model_list)


@views.route('/', methods=["POST"])
def predict_button():
    model_name = request.form.get('model')
    datasample = request.files['data-sample']
    modelpath = fr'.\models\{model_name}'
    evaluate_model.test(datasample, modelpath)
    return render_template("index.html", models=model_list)




