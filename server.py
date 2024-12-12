import streamlit as st
import pickle
from flask import Flask, request, jsonify
from scikit-learn
from pandas as pd

app = Flask(__name__)

model = pickle.load(open("movie_recommender_model.pkl", "rb"))

movies_df = pd.read_csv("movies.csv")
