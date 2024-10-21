import os
import math
from typing import List, Tuple, Any
from pyspark import SparkContext, SparkConf, RDD
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel
from time import time

# Increase the timeout for the Python worker
os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.python.worker.timeout=1200 pyspark-shell'

# Function to get absolute path from relative path
def get_abs_path(relative_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

# Function to load and process data
def load_data(sc: SparkContext, file_path: str, columns: List[int]) -> RDD[Tuple[Any]]:
    raw_data = sc.textFile(file_path)
    header = raw_data.take(1)[0]
    return raw_data.filter(lambda line: line != header) \
                   .map(lambda line: line.split(",")) \
                   .map(lambda tokens: tuple(tokens[i] for i in columns)) \
                   .cache()

# Function to print the first n rows of data
def print_rows(data: RDD[Tuple[Any]], n: int):
    for row in data.take(n):
        print(row)

# Function to train the ALS model and select the best rank
def train_model(train_RDD: RDD, val_RDD: RDD) -> int:
    ranks = [4, 8, 12, 16, 20]
    iterations = 10
    lambda_ = 0.1
    min_error = float('inf')
    best_rank = -1

    for rank in ranks:
        model = ALS.train(train_RDD, rank, seed=42, iterations=iterations, lambda_=lambda_)
        predictions = model.predictAll(val_RDD.map(lambda x: (x[0], x[1]))).map(lambda x: ((x[0], x[1]), x[2]))
        rates_and_preds = val_RDD.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean())
        print(f'For rank {rank} the RMSE is {error}')
        if error < min_error:
            min_error = error
            best_rank = rank

    return best_rank

# Function to predict ratings and calculate RMSE
def predict_and_evaluate(model, test_RDD: RDD) -> float:
    test_for_prediction = test_RDD.map(lambda x: (x[0], x[1]))
    predictions = model.predictAll(test_for_prediction).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    return math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

# Function to get counts and average ratings
def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1])) / nratings)

# Function to configure Spark
def configure_spark():
    conf = SparkConf().set("spark.python.worker.timeout", "1200")
    return SparkContext(conf=conf)

# Function to load and process movies data
def load_movies_data(sc: SparkContext, movies_file_path: str) -> RDD[Tuple[Any]]:
    movies_columns = [0, 1]
    return load_data(sc, movies_file_path, movies_columns)

# Function to load and process ratings data
def load_ratings_data(sc: SparkContext, ratings_file_path: str) -> RDD[Tuple[Any]]:
    ratings_columns = [0, 1, 2]
    return load_data(sc, ratings_file_path, ratings_columns)

# Function to split data into training, validation, and test sets
def split_data(ratings_data: RDD[Tuple[Any]]) -> Tuple[RDD, RDD, RDD]:
    train_RDD, val_RDD, test_RDD = ratings_data.randomSplit([6, 2, 2], seed=42)
    train_RDD.cache()
    val_RDD.cache()
    test_RDD.cache()
    return train_RDD, val_RDD, test_RDD

# Function to load the complete ratings dataset
def load_complete_ratings_data(sc: SparkContext, complete_ratings_file: str) -> RDD[Tuple[Any]]:
    return load_data(sc, complete_ratings_file, [0, 1, 2])

# Function to load the complete movies dataset
def load_complete_movies_data(sc: SparkContext, complete_movies_file: str) -> RDD[Tuple[Any]]:
    return load_data(sc, complete_movies_file, [0, 1, 2])

# Function to get movie ratings counts
def get_movie_ratings_counts(complete_ratings_data: RDD[Tuple[Any]]) -> RDD[Tuple[int, int]]:
    movie_ID_with_ratings_RDD = complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey()
    movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
    return movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

# Function to add new user ratings
def add_new_user_ratings(sc: SparkContext) -> RDD[Tuple[int, int, float]]:
    new_user_ratings = [
        (0, 260, 9),  # Star Wars (1977)
        (0, 1, 8),  # Toy Story (1995)
        (0, 16, 7),  # Casino (1995)
        (0, 25, 8),  # Leaving Las Vegas (1995)
        (0, 32, 9),  # Twelve Monkeys (a.k.a. 12 Monkeys) (1995)
        (0, 335, 4),  # Flintstones, The (1994)
        (0, 379, 3),  # Timecop (1994)
        (0, 296, 7),  # Pulp Fiction (1994)
        (0, 858, 10),  # Godfather, The (1972)
        (0, 50, 8)  # Usual Suspects, The (1995)
    ]
    return sc.parallelize(new_user_ratings)

# Function to train the new ratings model
def train_new_ratings_model(complete_ratings_data: RDD[Tuple[Any]], new_user_ratings_RDD: RDD[Tuple[int, int, float]], best_rank: int) -> MatrixFactorizationModel:
    complete_data_with_new_ratings_RDD = complete_ratings_data.union(new_user_ratings_RDD)
    t0 = time()
    new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=42, iterations=10, lambda_=0.1)
    tt = time() - t0
    print(f"New model trained in {round(tt, 3)} seconds")
    return new_ratings_model

# Function to get top recommendations
def get_top_recommendations(new_ratings_model: MatrixFactorizationModel, complete_movies_data: RDD[Tuple[Any]], movie_rating_counts_RDD: RDD[Tuple[int, int]], new_user_ratings_ids: List[int]) -> List[Tuple[str, float, int]]:
    new_user_ID = 0
    new_user_unrated_movies_RDD = complete_movies_data.filter(lambda x: x[0] not in new_user_ratings_ids).map(lambda x: (new_user_ID, x[0]))
    new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)
    new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
    complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))
    new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
    new_user_recommendations_rating_title_and_count_RDD = new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
    top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2] >= 25).takeOrdered(25, key=lambda x: -x[1])
    print('TOP recommended movies (with more than 25 reviews):\n%s' % '\n'.join(map(str, top_movies)))
    return top_movies

# Function to predict individual movie rating
def predict_individual_movie_rating(sc: SparkContext, new_ratings_model: MatrixFactorizationModel, movie_id: int) -> List[Tuple[int, int, float]]:
    my_movie = sc.parallelize([(0, movie_id)])
    individual_movie_rating_RDD = new_ratings_model.predictAll(my_movie)
    print(individual_movie_rating_RDD.take(1))
    return individual_movie_rating_RDD.take(1)

# Function to save and load the model
def save_and_load_model(sc: SparkContext, model: MatrixFactorizationModel):
    model_path = get_abs_path('models/movie_lens_als')
    model.save(sc, model_path)
    same_model = MatrixFactorizationModel.load(sc, model_path)
    return same_model

# Main function to run the code step-by-step
def main():
    # Step 1: Configure Spark
    sc = configure_spark()

    # Step 2: Define file paths
    movies_file_path = get_abs_path('data/ml-latest-small/cleaned_movies_small_df.csv')
    ratings_file_path = get_abs_path('data/ml-latest-small/cleaned_ratings_small_df.csv')
    complete_ratings_file = get_abs_path('data/ml-latest/ratings.csv')
    complete_movies_file = get_abs_path('data/ml-latest/movies.csv')

    # Step 3: Load and process movies data
    movies_data = load_movies_data(sc, movies_file_path)
    print_rows(movies_data, 3)

    # Step 4: Load and process ratings data
    ratings_data = load_ratings_data(sc, ratings_file_path)
    print_rows(ratings_data, 3)

    # Step 5: Split data into training, validation, and test sets
    train_RDD, val_RDD, test_RDD = split_data(ratings_data)

    # Step 6: Train the model and print the best rank
    best_rank = train_model(train_RDD, val_RDD)
    print(f'The best model was trained with rank {best_rank}')

    # Step 7: Train the model with the best rank and predict on the test set
    model = ALS.train(train_RDD, best_rank, seed=42, iterations=10, lambda_=0.1)
    error = predict_and_evaluate(model, test_RDD)
    print(f'For testing data the RMSE is {error}')

    # Step 8: Load the complete ratings dataset
    complete_ratings_data = load_complete_ratings_data(sc, complete_ratings_file)
    print(f"There are {complete_ratings_data.count()} recommendations in the complete dataset")

    # Step 9: Split the complete dataset into training and test sets
    training_RDD, test_RDD = complete_ratings_data.randomSplit([7, 3], seed=0)

    # Step 10: Train the complete dataset with the best rank
    complete_model = ALS.train(training_RDD, best_rank, seed=42, iterations=10, lambda_=0.1)
    error = predict_and_evaluate(complete_model, test_RDD)
    print(f'For testing data the RMSE is {error}')

    # Step 11: Load the complete movies dataset
    complete_movies_data = load_complete_movies_data(sc, complete_movies_file)
    complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))
    print(f"There are {complete_movies_titles.count()} movies in the complete dataset")

    # Step 12: Get movie ratings counts
    movie_rating_counts_RDD = get_movie_ratings_counts(complete_ratings_data)

    # Step 13: Add new user ratings
    new_user_ratings_RDD = add_new_user_ratings(sc)
    print(f'New user ratings: {new_user_ratings_RDD.take(10)}')

    # Step 14: Train the new ratings model
    new_ratings_model = train_new_ratings_model(complete_ratings_data, new_user_ratings_RDD, best_rank)

    # Step 15: Get top recommendations
    new_user_ratings_ids = list(map(lambda x: x[1], new_user_ratings_RDD.collect()))
    top_movies = get_top_recommendations(new_ratings_model, complete_movies_data, movie_rating_counts_RDD, new_user_ratings_ids)

    # Step 16: Predict individual movie rating
    predict_individual_movie_rating(sc, new_ratings_model, 500)  # Quiz Show (1994)

    # Step 17: Save and load the model
    save_and_load_model(sc, model)

if __name__ == "__main__":
    main()