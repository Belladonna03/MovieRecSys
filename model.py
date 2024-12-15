import os
import math
import logging
from typing import List, Tuple, Any
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
from time import time
from pyspark.rdd import RDD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SparkConfig:
    def __init__(self, timeout: int = 1200):
        self.timeout = timeout

    def get_spark_conf(self):
        conf = SparkConf().set("spark.python.worker.timeout", str(self.timeout))
        return conf

def get_abs_path(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

class MovieRecommender:
    def __init__(self, sc: SparkContext, rank: int = 10, iterations: int = 10, lambda_: float = 0.1):
        self.sc = sc
        self.rank = rank
        self.iterations = iterations
        self.lambda_ = lambda_

    def load_data(self, file_path: str, columns: List[int]) -> RDD[Tuple[Any]]:
        raw_data = self.sc.textFile(file_path)
        header = raw_data.take(1)[0]
        return raw_data.filter(lambda line: line != header) \
                       .map(lambda line: line.split(",")) \
                       .map(lambda tokens: tuple(tokens[i] for i in columns)) \
                       .cache()

    def split_data(self, ratings_data: RDD[Tuple[Any]]) -> Tuple[RDD, RDD, RDD]:
        train_RDD, val_RDD, test_RDD = ratings_data.randomSplit([6, 2, 2], seed=42)
        train_RDD.cache()
        val_RDD.cache()
        test_RDD.cache()
        return train_RDD, val_RDD, test_RDD

    def train_model(self, train_RDD: RDD, val_RDD: RDD) -> int:
        ranks = [4, 8, 12, 16, 20]
        min_error = float('inf')
        best_rank = -1

        for rank in ranks:
            model = ALS.train(train_RDD, rank, seed=42, iterations=self.iterations, lambda_=self.lambda_)
            predictions = model.predictAll(val_RDD.map(lambda x: (x[0], x[1]))).map(lambda x: ((x[0], x[1]), x[2]))
            rates_and_preds = val_RDD.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(predictions)
            error = math.sqrt(rates_and_preds.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean())
            logger.info(f'For rank {rank} the RMSE is {error}')
            if error < min_error:
                min_error = error
                best_rank = rank

        return best_rank

    def predict_and_evaluate(self, model: MatrixFactorizationModel, test_RDD: RDD) -> float:
        test_for_prediction = test_RDD.map(lambda x: (x[0], x[1]))
        predictions = model.predictAll(test_for_prediction).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        rmse = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
        logger.info(f'RMSE for testing data: {rmse}')
        return rmse

    def get_top_recommendations(self, model: MatrixFactorizationModel, complete_movies_data: RDD[Tuple[Any]], movie_rating_counts_RDD: RDD[Tuple[int, int]], new_user_ratings_ids: List[int]) -> List[Tuple[str, float, int]]:
        new_user_ID = 0
        new_user_unrated_movies_RDD = complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0]))
        new_user_recommendations_RDD = model.predictAll(new_user_unrated_movies_RDD)
        new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
        complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))
        new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
        new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
        top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2] >= 25).takeOrdered(25, key=lambda x: -x[1])
        logger.info(f'TOP recommended movies (with more than 25 reviews):\n{top_movies}')
        return top_movies

def main():
    spark_config = SparkConfig(timeout=1200)
    sc = SparkContext(conf=spark_config.get_spark_conf())
    
    movies_file_path = get_abs_path('data/ml-latest-small/cleaned_movies_small_df.csv')
    ratings_file_path = get_abs_path('data/ml-latest-small/cleaned_ratings_small_df.csv')
    complete_ratings_file = get_abs_path('data/ml-latest/ratings.csv')
    complete_movies_file = get_abs_path('data/ml-latest/movies.csv')

    recommender = MovieRecommender(sc)
    movies_data = recommender.load_data(movies_file_path, [0, 1])
    ratings_data = recommender.load_data(ratings_file_path, [0, 1, 2])
    
    logger.info(f"Loaded {movies_data.count()} movies and {ratings_data.count()} ratings.")

    train_RDD, val_RDD, test_RDD = recommender.split_data(ratings_data)

    best_rank = recommender.train_model(train_RDD, val_RDD)
    logger.info(f'The best model was trained with rank {best_rank}')

    model = ALS.train(train_RDD, best_rank, seed=42, iterations=10, lambda_=0.1)
    rmse = recommender.predict_and_evaluate(model, test_RDD)

    complete_movies_data = recommender.load_data(complete_movies_file, [0, 1, 2])
    complete_ratings_data = recommender.load_data(complete_ratings_file, [0, 1, 2])

    new_user_ratings_ids = [260, 1, 16, 25, 32, 335, 379, 296, 858, 50]  # IDs фильмов, оцененных новым пользователем
    movie_rating_counts_RDD = complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey().map(lambda x: (x[0], len(x[1])))
    top_movies = recommender.get_top_recommendations(model, complete_movies_data, movie_rating_counts_RDD, new_user_ratings_ids)

    logger.info("Recommendation process completed successfully.")

if __name__ == "__main__":
    main()
