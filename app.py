from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load pipeline (scaler + model)
pipeline = joblib.load('model/fish_pipeline.pkl')
encoder = joblib.load('model/species_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs from form
    weight = float(request.form['Weight'])
    length1 = float(request.form['Length1'])
    length2 = float(request.form['Length2'])
    length3 = float(request.form['Length3'])
    height = float(request.form['Height'])
    width = float(request.form['Width'])

    input_df = pd.DataFrame([{
        'Weight': weight,
        'Length1': length1,
        'Length2': length2,
        'Length3': length3,
        'Height': height,
        'Width': width
    }])

    # Predict directly using pipeline
    prediction = pipeline.predict(input_df)
    
    # If your target is encoded:
    species = encoder.inverse_transform(prediction)[0]  # <-- only if label encoded

    return render_template('index.html', prediction_text=f'Predicted Species: {species}')

if __name__ == '__main__':
    app.run(debug=True)
