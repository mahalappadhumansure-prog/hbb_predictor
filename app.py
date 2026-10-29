from flask import Flask, Response, jsonify, render_template, request

import model

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/examples")
def examples():
    return jsonify(model.EXAMPLES)


@app.get("/api/model-info")
def model_info():
    return jsonify({"metrics": model.METRICS, "global_importance": model.GLOBAL_IMPORTANCE})


@app.get("/summary.png")
def summary():
    return Response(model.SUMMARY_PNG, mimetype="image/png")


@app.post("/api/predict")
def predict():
    text = (request.get_json(silent=True) or {}).get("sequence", "")
    try:
        return jsonify(model.analyze(text))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
