from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)
pipeline = PredictPipeline()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle CSV file upload
        file = request.files.get("file")
        if file and file.filename != "":
            if file.filename.endswith(".csv"):
                try:
                    df = pd.read_csv(file)
                    predictions = pipeline.predict(df)
                    df["Predicted Carbon Footprint"] = predictions
                    return render_template(
                        "index.html",
                        tables=[df.to_html(classes="table table-striped", index=False)],
                        csv_uploaded=True
                    )
                except Exception as e:
                    return render_template("index.html", error=f"Error processing CSV: {str(e)}")
            else:
                return render_template("index.html", error="Please upload a valid CSV file.")
        else:
            # Handle form submission
            form_data = {}
            for field, value in request.form.items():
                # Try to convert numeric fields
                try:
                    form_data[field] = float(value)
                except ValueError:
                    form_data[field] = value

            input_df = pd.DataFrame([form_data])

            try:
                predictions = pipeline.predict(input_df)
                return render_template("index.html", prediction=predictions[0], csv_uploaded=False)
            except Exception as e:
                return render_template("index.html", error=f"Error during prediction: {str(e)}")

    # Default GET request
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
