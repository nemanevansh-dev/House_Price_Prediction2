from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    dtc_model = pickle.load(open('dtc_trained_model.pkl', 'rb'))
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Model file not found. Please run model.py first!")
    dtc_model = None


@app.route('/', methods=['GET', 'POST'])
def predict_loan_default():
    if request.method == 'GET':
        return render_template('loan_default_predictor.html')

    elif request.method == 'POST':
        if dtc_model is None:
            return render_template('prediction_result.html',
                                   prediction="Error: Model not loaded. Please train the model first.")

        try:
            # Get form data and convert to integers
            gender = 1 if request.form['gender'] == 'Male' else 0
            age = int(request.form['age'])
            income = int(request.form['income'])
            loan_amt = int(request.form['loan_amt'])
            loan_term = int(request.form['loan_term'])
            credit_score = int(request.form['credit_score'])
            emp_status = int(request.form['emp_status'])
            marital_status = int(request.form['marital_status'])
            prev_defaults = int(request.form['prev_defaults'])

            # Create feature array
            features = np.array([[gender, age, income, loan_amt, loan_term,
                                  credit_score, emp_status, marital_status, prev_defaults]])

            # Make prediction
            prediction = dtc_model.predict(features)[0]
            prediction_proba = dtc_model.predict_proba(features)[0]

            if prediction == 0:
                result = "✅ LOW RISK: The person is not likely to default on the loan"
                confidence = prediction_proba[0]
            else:
                result = "⚠️ HIGH RISK: The person is likely to default on the loan"
                confidence = prediction_proba[1]

            return render_template('prediction_result.html',
                                   prediction=result,
                                   confidence=f"{confidence:.2%}")

        except Exception as e:
            return render_template('prediction_result.html',
                                   prediction=f"Error processing request: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)