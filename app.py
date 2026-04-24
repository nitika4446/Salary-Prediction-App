import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Employee Salary Prediction App")
st.write("Predict salary based on years of experience using Machine Learning.")

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("salary_data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# Data Visualization
# ---------------------------
st.subheader("Experience vs Salary Visualization")

fig, ax = plt.subplots()
ax.scatter(df["YearsExperience"], df["Salary"])
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Salary vs Experience")
st.pyplot(fig)

# ---------------------------
# Model Training
# ---------------------------
X = df[["YearsExperience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"R² Score: {r2:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f}")

# ---------------------------
# User Input
# ---------------------------
st.subheader("Predict Salary")

experience = st.slider(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1
)

if st.button("Predict Salary"):
    prediction = model.predict([[experience]])
    
    st.success(
        f"Estimated Salary for {experience} years experience: ₹{prediction[0]:,.2f}"
    )

# ---------------------------
# Regression Line
# ---------------------------
st.subheader("Regression Line")

fig2, ax2 = plt.subplots()

ax2.scatter(X, y, color="blue")
ax2.plot(X, model.predict(X), color="red")

ax2.set_xlabel("Years of Experience")
ax2.set_ylabel("Salary")
ax2.set_title("Linear Regression Fit")

st.pyplot(fig2)

# Footer
st.markdown("---")
st.write("Built with ❤️ using Streamlit + Machine Learning")

