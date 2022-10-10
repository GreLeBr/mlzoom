import pickle

# from flask import Flask
# from flask import request
# from flask import jsonify


model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)   
    
def predict(customer):
    # customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    price = y_pred >= 0.5

    result = {
        'price_probability': float(y_pred),
        'price': bool(price)
    }
    return print(result)
    # return jsonify(result)


if __name__ == "__main__":
    customer = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
    predict(customer)
    # app.run(debug=True, host='0.0.0.0', port=9696)    