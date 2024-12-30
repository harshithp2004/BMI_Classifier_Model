from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

df=pd.read_csv("C:/Users/Harshith/Downloads/bmi.csv")
label_encoder=LabelEncoder()
df['BmiClass']=label_encoder.fit_transform(df['BmiClass'])

x=df[['Age','Height','Weight']]
y=df['BmiClass']
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=42)

model=RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        age = float(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        prediction = model.predict([[age, height, weight]])
        result = label_encoder.inverse_transform(prediction)[0]
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
