import flask
import pickle

from CONFIG import PASSWORD_COMPLEXITY_MODEL_PATH
from ml_models.password_complexity.main import get_features

with open(PASSWORD_COMPLEXITY_MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__)

@app.route('/predict-password-complexity-v1', methods=['GET', 'POST'])
def predict_password_complexity_v1():
    if flask.request.method in ['POST', 'GET']:
        submited_password = flask.request.values['password']
   
        input_features = get_features(submited_password)
        prediction = model.predict(input_features)[0]
        return {"prediction": round(float(prediction), 3)}

if __name__ == '__main__':
    app.run(debug=True)