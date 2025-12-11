from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))
iris_labels = ["Setosa", "Versicolor", "Virginica"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred = model.predict(data)[0]
    predicted_class = iris_labels[pred]

    return render_template("result.html", result=predicted_class)

if __name__ == "__main__":
    app.run(debug=True)
