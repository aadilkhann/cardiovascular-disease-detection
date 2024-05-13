from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.svm import SVC
import joblib

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("preprocessed_dataset.csv")
X = df[['age', 'gender', 'height', 'weight', 'systolic', 'diastolic', 'cholesterol', 'glucose', 'smoke', 'alcohol', 'active', 'pulse_pressure']]
y = df['cardiovascular_disease']

# Check if the model is already trained and saved
try:
    # Load the pre-trained model
    clf = joblib.load("svm_model.pkl")
except FileNotFoundError:
    # If the model is not found, train and save it
    clf = SVC(kernel='rbf', C=5)
    clf.fit(X, y)
    joblib.dump(clf, "svm_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        gender = int(request.form['gender'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        systolic = float(request.form['systolic'])
        diastolic = float(request.form['diastolic'])
        cholesterol = int(request.form['cholesterol'])
        glucose = int(request.form['glucose'])
        smoke = int(request.form['smoke'])
        alcohol = int(request.form['alcohol'])
        active = int(request.form['active'])
        pulse_pressure = systolic - diastolic

        input_data = [[age, gender, height, weight, systolic, diastolic, cholesterol, glucose, smoke, alcohol, active, pulse_pressure]]

        # Perform prediction
        prediction = clf.predict(input_data)[0]

        # Redirect to loading page while model is predicting
        return redirect(url_for('result', prediction=prediction))

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/result/<prediction>')
def result(prediction):
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
