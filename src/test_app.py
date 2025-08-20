from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "Flask is working!"

if __name__ == "__main__":
    app.run(port=8000, debug=True)
