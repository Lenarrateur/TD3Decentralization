from flask import Flask, request, jsonify
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Load Iris dataset and train the model
data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        features = [float(x) for x in request.args.getlist('features')]        
        probabilities = model.predict_proba([features])[0]
        ##prediction = np.argmax(probabilities)
        return jsonify({
            #'prediction': int(prediction),
            'probabilities': probabilities.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

