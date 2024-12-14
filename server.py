from flask import Flask, request, jsonify
import pickle
import logging
from model import MovieRecommender  # Импортируем класс из вашего кода с моделью

# Инициализация Flask
app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели и данных
def load_model():
    try:
        # Загрузка обученной модели (можно использовать pickle или иной способ сохранения)
        model = pickle.load(open("movie_recommender_model.pkl", "rb"))
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

# Главная страница
@app.route('/')
def home():
    return "Welcome to the Movie Recommender Service!"

# API для получения рекомендаций
@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        genre = request.args.get('genre')
        actor = request.args.get('actor')

        if not genre or not actor:
            return jsonify({"error": "Both genre and actor parameters are required"}), 400

        # Получаем модель
        model = load_model()
        if not model:
            return jsonify({"error": "Model could not be loaded"}), 500

        # Получение рекомендаций через модель
        recommendations = model.get_top_recommendations(genre, actor)

        if not recommendations:
            return jsonify({"error": "No recommendations found for the given inputs"}), 404

        # Форматирование рекомендаций для ответа
        recommendations_data = [{"movie": rec[0], "rating": rec[1]} for rec in recommendations]

        return jsonify(recommendations_data)

    except Exception as e:
        logger.error(f"Error in recommendation process: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

# Запуск Flask-сервера
if __name__ == "__main__":
    app.run(debug=True)
