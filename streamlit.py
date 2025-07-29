import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("heart.csv")

st.title("📊 Heart Disease Prediction Dashboard")

st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to", ["EDA", "Modeling", "Predict Heart Disease"])

if section == "EDA":
    st.subheader("Exploratory Data Analysis")

    fig1 = px.histogram(df, x='Age', nbins=20, title='Age Distribution',
                        color_discrete_sequence=['lightcoral'])
    st.plotly_chart(fig1)

    fig2 = px.histogram(df, x='ChestPainType', color='HeartDisease', barmode='group',
                        title='Chest Pain Type vs Heart Disease',
                        color_discrete_sequence=['skyblue', 'lightcoral'])
    st.plotly_chart(fig2)

    fig3 = px.box(df, x='HeartDisease', y='MaxHR', title='Max Heart Rate vs Heart Disease')
    st.plotly_chart(fig3)

    fig4 = px.histogram(df, x='FastingBS', color='HeartDisease', barmode='group',
                        title='Fasting Blood Sugar vs Heart Disease',
                        color_discrete_sequence=['skyblue', 'lightcoral'])
    st.plotly_chart(fig4)

    fig5 = px.scatter_matrix(df,
        dimensions=['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak'],
        color='HeartDisease', title='Pairplot of Numerical Features')
    st.plotly_chart(fig5)

elif section == "Modeling":
    st.subheader("Model Performance Comparison")

    df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear'),
        'Random Forest': RandomForestClassifier(n_estimators=100, criterion="entropy", max_depth=4, random_state=0),
        'SVM': SVC(gamma='scale')
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)

        st.write(f"### {name}")
        st.write(f"**Accuracy:** {acc * 100:.2f}%")
        st.write(f"**Recall:** {rec:.2f}")
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

elif section == "Predict Heart Disease":
    st.subheader("Make a Prediction")
    st.write("Enter patient data to predict risk of heart disease.")

    # Prepare the same encoding and scaling as in the Modeling section
    df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)
    X = df_encoded.drop('HeartDisease', axis=1)
    y = df_encoded['HeartDisease']
    scaler = StandardScaler()
    scaler.fit(X)

    input_data = {}
    input_data['Age'] = st.number_input("Age", 20, 100, 50)
    input_data['RestingBP'] = st.number_input("Resting Blood Pressure", 80, 200, 120)
    input_data['Cholesterol'] = st.number_input("Cholesterol", 100, 400, 200)
    input_data['FastingBS'] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    input_data['MaxHR'] = st.number_input("Max Heart Rate", 60, 220, 150)
    input_data['Oldpeak'] = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    input_data['Sex_M'] = st.selectbox("Sex", ["F", "M"]) == "M"
    input_data['ChestPainType_ATA'] = st.selectbox("Chest Pain Type", ["NAP", "ATA", "ASY", "TA"] ) == "ATA"
    input_data['ChestPainType_NAP'] = input_data['ChestPainType_ATA'] == False and input_data['ChestPainType_ATA'] == False
    input_data['ChestPainType_TA'] = st.selectbox("Secondary Chest Pain Type", ["ASY", "TA"]) == "TA"
    input_data['RestingECG_Normal'] = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"]) == "Normal"
    input_data['RestingECG_ST'] = st.selectbox("Secondary ECG Type", ["LVH", "ST"]) == "ST"
    input_data['ExerciseAngina_Y'] = st.selectbox("Exercise-induced Angina", ["Y", "N"]) == "Y"
    input_data['ST_Slope_Flat'] = st.selectbox("ST Slope", ["Flat", "Up", "Down"]) == "Flat"
    input_data['ST_Slope_Up'] = input_data['ST_Slope_Flat'] == False and input_data['ST_Slope_Flat'] == False

    input_df = pd.DataFrame([input_data])
    input_df = input_df.astype(float)

    # Ensure input_df has all columns as X, fill missing with 0
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X.columns]

    input_df_scaled = scaler.transform(input_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(solver='liblinear')
    model.fit(scaler.transform(X_train), y_train)
    prediction = model.predict(input_df_scaled)[0]

    st.write("### Prediction Result")
    if prediction == 1:
        st.markdown(
            "<span style='color:red; font-weight:bold;'>🚨 The model predicts: HEART DISEASE PRESENT</span>",
            unsafe_allow_html=True
        )
    else:
        st.success("The model predicts: NO HEART DISEASE")
