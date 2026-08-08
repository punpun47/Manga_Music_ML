from flask import Flask, request, Response
import requests

app = Flask(__name__)

@app.route("/fetch-image")
def fetch_image():
    url = request.args.get("url")
    resp = requests.get(url)
    return Response(resp.content, mimetype="image/png")


if __name__ == "__main__":
    app.run(port=5001)