""" Script to prevent the bot entering sleep state.

We will be pining the url of the flask server from uptime robot for every 5min to keep the bot alive. So, even if you close this Repl file, the script will still be running.
"""


from flask import Flask
from threading import Thread

app = Flask('')

@app.route('/')
def home():
    return "Your bot is currently active"

def run():
  app.run(host='0.0.0.0',port=8080)

def bot_status():
    t = Thread(target=run)
    t.start()