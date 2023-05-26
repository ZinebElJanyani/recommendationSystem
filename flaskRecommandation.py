from flask import Flask, jsonify, request
from model import get_top_recommendations

app = Flask(__name__)

@app.route('/recommendations/<user_id>')
def recommendations(user_id):
    user_id = int(user_id)
    recommended_products = get_top_recommendations(user_id)
    recommended_products = [int(x) for x in recommended_products]
    return jsonify(recommended_products)

if __name__ == '__main__':
    app.run(debug=True)