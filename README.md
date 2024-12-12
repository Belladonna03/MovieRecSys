## 🎬 **Movie Recommender System** with Apache Spark 🚀

Welcome to the **Movie Recommendation System** built with **Apache Spark**! This powerful system utilizes the **Alternating Least Squares (ALS)** algorithm to recommend movies to users based on their preferences and ratings.

---

## ✨ **Features**

- **Data Processing**: Fast and efficient data processing using **Spark RDDs**, tailored for large-scale movie recommendation tasks.
- **ALS Model**: Leveraging the **ALS** algorithm for matrix factorization to predict ratings and make movie recommendations.
- **Data Splitting**: Split datasets into **training**, **validation**, and **test** sets for rigorous model evaluation.
- **Top Recommendations**: Provides **top movie recommendations** for a new user based on personalized ratings.
- **Logging**: Includes **detailed logging** to monitor the process and output during training and recommendation generation.
- **Modular Code**: Structured to easily extend and maintain, adding new features and improvements.

---

## 📋 **Requirements**

To run this project, make sure you have the following:

- **Apache Spark**: The core distributed computing framework for processing large datasets.
- **Python 3.x**: Python is used for scripting and model training.
- **PySpark**: The Python API for Spark, enabling integration with Spark functionalities.
- **Java 8+**: Required to run Spark.
- **Dependencies**: Install necessary Python libraries via `requirements.txt`.

---

## 🛠️ **Setup and Installation**

Follow these steps to get started:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/movie-recommender-system.git
   cd movie-recommender-system
   ```

2. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Apache Spark**:
   - If you haven't installed Spark, follow the [installation guide](https://spark.apache.org/docs/latest/).
   - Make sure to configure the `SPARK_HOME` environment variable and add Spark binaries to your `PATH`.

---

## 🚀 **Running the Application**

### **Step 1**: Prepare your datasets:
You need two CSV datasets:
- **Movies Dataset**: Contains movie IDs and titles.
- **Ratings Dataset**: Contains user IDs, movie IDs, and ratings.

Example paths for these files:
- `data/ml-latest-small/cleaned_movies_small_df.csv`
- `data/ml-latest-small/cleaned_ratings_small_df.csv`

### **Step 2**: Run the `main.py` script:
```bash
python main.py
```

This will:
- Load and process the data.
- Train the **ALS** model using the ratings data.
- Split the data into **training**, **validation**, and **test** sets.
- Evaluate the model's performance using **RMSE**.
- Generate **top movie recommendations** for a new user.
- Save the trained model for future use.

---

## 📊 **Example Output**

When the application runs, you’ll see output like this in the terminal:

```
Loaded 1000 movies and 5000 ratings.
For rank 4 the RMSE is 0.936
For rank 8 the RMSE is 0.874
The best model was trained with rank 8
RMSE for testing data: 0.856
TOP recommended movies (with more than 25 reviews):
[('Star Wars (1977)', 8.9, 30), ('The Matrix (1999)', 8.7, 50), ...]
```

---

## 📈 **Logging**

With **Python's `logging` module**, you can track:
- Training progress.
- RMSE for different ranks.
- Top recommendations generated for users.

Logs are output to the terminal, providing insights into the entire process.

---

## 🤝 **Contributing**

We welcome your contributions to improve and extend this project! Here’s how you can contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to your forked repository (`git push origin feature-branch`).
5. Create a pull request to the `main` branch.

---

## 📝 **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 🙏 **Acknowledgments**

- Special thanks to the **Apache Spark** team for building a powerful distributed computing engine.
- This project is based on the **ALS** algorithm, which is a widely used collaborative filtering technique in recommendation systems.
