from flask import Blueprint, render_template, request
import os
import predict

views = Blueprint(__name__, 'views')
model_list = os.listdir(r'.\models')


@views.route("/")
def home():
    button_clicked = False
    return render_template("index.html", models=model_list, button_clicked=button_clicked)


@views.route('/', methods=["POST"])
def predict_button():
    model_name = request.form.get('model')
    datasample = request.files['data-sample']
    modelpath = fr'.\models\{model_name}'
    date, pred = predict.predict(datasample, modelpath)
    button_clicked = True
    return render_template("index.html",
                           models=model_list,
                           abnormality_prob=pred,
                           button_clicked=button_clicked,
                           date=date)
