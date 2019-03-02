from flask import Flask, render_template
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pusher import Pusher
import requests, json, atexit, time, plotly, plotly.graph_objs as go

# create flask app
app = Flask(__name__)

# configure pusher object
pusher = Pusher(
    app_id='726413',
    key='f26ddd693c9477bad4c9',
    secret='e2c955adb821315bfff2',
    cluster='us2',
    ssl=True
)

# define variables for data retrieval
times = []
currencies = ["BTC"]
prices = {"BTC": []}

@app.route("/")
def index():
    return render_template("index.html")
