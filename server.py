from flask import Flask, request, jsonify
import pickle
import logging
from model import MovieRecommender

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    try:
        model = pickle.load(open("movie_recommender_model.pkl", "rb"))
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

@app.route('/')
def home():
    return "Welcome to the Movie Recommender Service!"

@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        genre = request.args.get('genre')
        actor = request.args.get('actor')

        if not genre or not actor:
            return jsonify({"error": "Both genre and actor parameters are required"}), 400

        model = load_model()
        if not model:
            return jsonify({"error": "Model could not be loaded"}), 500

        recommendations = model.get_top_recommendations(genre, actor)

        if not recommendations:
            return jsonify({"error": "No recommendations found for the given inputs"}), 404

        recommendations_data = [{"movie": rec[0], "rating": rec[1]} for rec in recommendations]

        return jsonify(recommendations_data)

    except Exception as e:
        logger.error(f"Error in recommendation process: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == "__main__":
    app.run(debug=True)
