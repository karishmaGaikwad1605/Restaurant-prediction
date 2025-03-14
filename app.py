import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, template_folder='template')

# Load trained model
Model = pickle.load(open('Model2.pkl', 'rb'))

# Define expected column names (from training data)
expected_columns = ['Menu', 'Delivery', 'Booking', 'No_of_Best_Sellers', 'No_of_Varieties',
       'Cost_Per_Person', 'Price_Category', 'Average', 'Excellent', 'Good',
       'Poor', 'BTM', 'Banashankari', 'Bannerghatta Road', 'Basavanagudi',
       'Bellandur', 'Brigade Road', 'Brookefield', 'Church Street',
       'Electronic City', 'Frazer Town', 'HSR', 'Indiranagar', 'JP Nagar',
       'Jayanagar', 'Kalyan Nagar', 'Kammanahalli', 'Koramangala 4th Block',
       'Koramangala 5th Block', 'Koramangala 6th Block',
       'Koramangala 7th Block', 'Lavelle Road', 'MG Road', 'Malleshwaram',
       'Marathahalli', 'New BEL Road', 'Old Airport Road', 'Rajajinagar',
       'Residency Road', 'Sarjapur Road', 'Whitefield']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input values
        data = {
            "Online_Order": int(request.form["Online_Order"]),
            "Book_Table": int(request.form["Book_Table"]),
            "Menu": int(request.form["Menu"]),
            "City": request.form["City"],
            "No_of_Varities": int(request.form["No_of_Varities"]),
            "Category": request.form["Category"],
            "Cost_Per_Person": float(request.form["Cost_Per_Person"])
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Convert categorical features into one-hot encoding (same as training)
        df_Category_dummies = pd.get_dummies(df['Category'], dummy_na=False)
        df_City_dummies = pd.get_dummies(df['City'], dummy_na=False)

        df = pd.concat([df, df_Category_dummies, df_City_dummies], axis=1)

        # Drop original categorical columns
        df.drop(['Category', 'City'], axis=1, inplace=True)

        # Ensure feature order and fill missing values with 0
        df = df.reindex(columns=expected_columns, fill_value=0)

        # Convert to NumPy array
        final_features = df.to_numpy()

        # Make prediction
        prediction = Model.predict(final_features)

        # Round output
        output = round(prediction[0], 1)

        return render_template('index.html', prediction_text='Your Rating is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
