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
    model = pickle.load(open("movie_recommender_model.pkl", "rb"))  # Загрузка модели
    return model

# Главная страница
@app.route('/')
def home():
    return "Welcome to the Movie Recommender Service!"

# API для получения рекомендаций
@app.route('/recommend', methods=['GET'])
def recommend():
    genre = request.args.get('genre')
    actor = request.args.get('actor')

    # Получение рекомендаций через модель
    model = load_model()
    recommendations = model.get_top_recommendations(genre, actor)

    # Форматирование рекомендаций для ответа
    recommendations_data = [{"movie": rec[0], "rating": rec[1]} for rec in recommendations]
    
    return jsonify(recommendations_data)

# Функция для получения рекомендаций на основе жанра и актера
def get_recommendations(genre, actor):
    # Логика получения рекомендаций на основе выбранных данных
    recommendations = []  # Тут будет ваш код для генерации рекомендаций с использованием модели
    return recommendations

# Запуск Flask-сервера
if __name__ == "__main__":
    app.run(debug=True)
