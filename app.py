import io
import base64
import matplotlib

matplotlib.use("Agg")  # Set the backend to use the "Agg" renderer
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
import pandas as pd
import seaborn as sns
from flask_cors import CORS
import joblib
import sklearn
import pickle

app = Flask(_name_)
cors = CORS(app)



def generate_graphs():
    df = pd.read_csv("data.csv")
    # Perform data processing and generate graphs
    df["CrimeTime"] = df["CrimeTime"].str.replace(":", "")
    df["Date"] = df["CrimeDate"] + " " + df["CrimeTime"]
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H%M", errors="coerce")
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["Weekday"] = df["Date"].dt.weekday + 1
    df["Hour"] = df["Date"].dt.hour

    # Generate multiple graphs
    graphs = []

    # Graph 1
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(x="Month", data=df)
    plt.ylabel("Crime Frequency", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xlabel("Month", fontfamily="Arial", fontsize=25, fontweight="bold")
    labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    ax.set_xticklabels(labels)
    plt.xticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.yticks(fontfamily="Arial", fontsize=20, fontweight="bold")

    plt.plot()
    graph1_stream = io.BytesIO()
    plt.savefig(graph1_stream, format="png")
    plt.close()  # Close the figure after saving
    graph1_stream.seek(0)
    graphs.append(
        {
            "image": base64.b64encode(graph1_stream.getvalue()).decode("utf-8"),
            "des": "Frequency of Crime by Month",
        }
    )

    # Graph 2
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(x="Weapon", data=df)
    plt.ylabel("Crime Frequency", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xlabel("Weapon Used", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.yticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.plot()
    graph2_stream = io.BytesIO()
    plt.savefig(graph2_stream, format="png")
    plt.close()  # Close the figure after saving
    graph2_stream.seek(0)
    graphs.append(
        {
            "image": base64.b64encode(graph2_stream.getvalue()).decode("utf-8"),
            "des": "Frequency of Crime by Weapon Used",
        }
    )

    # Graph 3
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(x="District", data=df)
    plt.ylabel("Crime Frequency", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xlabel("District", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xticks(fontfamily="Arial", fontsize=15, fontweight="bold", rotation=19)
    plt.yticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.plot()
    graph3_stream = io.BytesIO()
    plt.savefig(graph3_stream, format="png")
    plt.close()  # Close the figure after saving
    graph3_stream.seek(0)
    graphs.append(
        {
            "image": base64.b64encode(graph3_stream.getvalue()).decode("utf-8"),
            "des": "Frequency of Crime by District",
        }
    )

    # Graph 4
    # plt.figure(figsize=(9, 4), dpi=80)
    plt.figure(figsize=(12, 8))

    ax = sns.countplot(x="Year", data=df)
    plt.ylabel("Crime Frequency", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xlabel("Year", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.yticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.plot()
    graph4_stream = io.BytesIO()
    plt.savefig(graph4_stream, format="png")
    plt.close()  # Close the figure after saving
    graph4_stream.seek(0)
    graphs.append(
        {
            "image": base64.b64encode(graph4_stream.getvalue()).decode("utf-8"),
            "des": "Frequency of Crime by Year",
        }
    )

    # Graph 5
    plt.figure(figsize=(12, 8))
    ax = sns.countplot(x="Weekday", data=df)
    plt.ylabel("Crime Frequency", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xlabel("Day of Week", fontfamily="Arial", fontsize=25, fontweight="bold")
    labels = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
    ax.set_xticklabels(labels)
    plt.xticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.yticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.plot()
    graph5_stream = io.BytesIO()
    plt.savefig(graph5_stream, format="png")
    plt.close()  # Close the figure after saving
    graph5_stream.seek(0)
    graphs.append(
        {
            "image": base64.b64encode(graph5_stream.getvalue()).decode("utf-8"),
            "des": "Frequency of Crime by Day of Week",
        }
    )

    # Graph 6
    plt.figure(figsize=(12, 7))
    ax = sns.countplot(x="Year", hue="Weapon", data=df)
    plt.ylabel("Crime Frequency", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xlabel("Year", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.yticks(fontfamily="Arial", fontsize=20, fontweight="bold")
    plt.plot()
    graph7_stream = io.BytesIO()
    plt.savefig(graph7_stream, format="png")
    plt.close()  # Close the figure after saving
    graph7_stream.seek(0)
    graphs.append(
        {
            "image": base64.b64encode(graph7_stream.getvalue()).decode("utf-8"),
            "des": "Frequency of Crime per Year Grouped by Weapon Used",
        }
    )

    # Graph 7
    plt.figure(figsize=(17, 10))
    ax = sns.countplot(x="Hour", data=df)
    plt.ylabel("Crime Frequency", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xlabel("Hour", fontfamily="Arial", fontsize=25, fontweight="bold")
    plt.xticks(fontfamily="Arial", fontsize=22, fontweight="bold", rotation=90)
    plt.yticks(fontfamily="Arial", fontsize=22, fontweight="bold")
    plt.plot()
    graph6_stream = io.BytesIO()
    plt.savefig(graph6_stream, format="png")
    plt.close()  # Close the figure after saving
    graph6_stream.seek(0)
    graphs.append(
        {
            "image": base64.b64encode(graph6_stream.getvalue()).decode("utf-8"),
            "des": "Frequency of Crime by Hour of Day",
        }
    )

    # Return the generated graphs
    return graphs


@app.route("/api/graphs", methods=["GET"])
def get_graphs():
    graphs = generate_graphs()
    return jsonify(graphs)


# Load the saved model
# knn = joblib.load('D:\FYP_WEB_INTERFACE\serverFlask\backend\crime_model.joblib')

@app.route('/predict_crime_knn', methods=['POST'])
def predict_crime_knn():
    with open('crime_model.pkl', 'rb') as file:
        knn = pickle.load(file)

        data = request.get_json()

        longitude = float(data['longitude'])
        latitude = float(data['latitude'])

        day = int(data['day'])
        month = int(data['month'])
        year = int(data['year'])

        # Prepare the input data for prediction
        X_input = [[longitude, latitude, day, month, year]]

        # Predict the total number of crimes
        prediction = knn.predict(X_input)
        serialized_data = prediction.tolist()
        # Return the prediction as a JSON response
        return jsonify({'prediction': serialized_data})


@app.route('/predict_crime_random_forest', methods=['POST'])
def predict_crime_random_forest():
    with open('crime_model_random_forest.pkl', 'rb') as file:
        rf = pickle.load(file)

        data = request.get_json()
        print(data)

        longitude = float(data['longitude'])
        latitude = float(data['latitude'])

        day = int(data['day'])
        month = int(data['month'])
        year = int(data['year'])

        # Prepare the input data for prediction
        X_input = [[longitude, latitude, day, month, year]]

        # Predict the total number of crimes
        prediction = rf.predict(X_input)
        print(prediction)
        # serialized_data = prediction.tolist()
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction[0]})


if _name_ == "_main_":
    app.run(port=8888)