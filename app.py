
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="Loan Approval Prediction",
    layout="wide"
)


st.sidebar.title("Loan Approval System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Details",
        "Dataset",
        "EDA",
        "Model Comparison",
        "Prediction"
    ]
)


# Load Machine Learning Models
models = {
    "Logistic Regression": joblib.load("logistic_model.joblib"),
    "KNN": joblib.load("knn_model.joblib"),
    "SVC": joblib.load("svc_model.joblib"),
    "Best Model (SVC)": joblib.load("best_model.joblib")
}


# Load Model Comparison
comparison = pd.read_csv("model_comparison.csv")


if page == "Project Details":

    st.title("Loan Approval Prediction System")

    st.write(
        "A Machine Learning Classification Project"
    )

    st.divider()

    st.subheader("Project Overview")

    st.write(
        "This project aims to predict whether a loan application "
        "will be approved or rejected using applicant details such as "
        "income, credit score, loan amount, employment experience, and city."
    )

    st.subheader("Project Purpose")

    st.write(
        "Evaluating loan applications manually can be time-consuming. "
        "This project uses machine learning to predict whether a loan "
        "application will be approved or rejected based on factors such "
        "as income, credit score, loan amount, employment experience, and city."
    )

    st.subheader("Dataset Information")

    st.write(
        "The dataset contains applicant information including "
        "income, credit score, loan amount, years employed and city."
    )

    st.write("**Target Variable:** Loan Approval")

    st.write("""
    **Target Meaning:**
    - False = Loan Not Approved
    - True = Loan Approved
    """)

    st.subheader("Machine Learning Algorithms")

    st.markdown("""
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Support Vector Classifier (SVC)
    """)


elif page == "Dataset":

    st.title("Loan Approval Dataset")

    df = pd.read_csv("loan_approval.csv")

    st.write("Raw Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Dataset Information")

    st.write("Total Records:", df.shape[0])
    st.write("Total Columns:", df.shape[1])


elif page == "EDA":

    st.title("Exploratory Data Analysis")

    st.write(
        "Analysis and visualizations of the raw loan approval dataset."
    )

    df = pd.read_csv("loan_approval.csv")

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("Missing Values Analysis")

    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=missing_df,
        x="Column",
        y="Missing Values",
        ax=ax
    )

    ax.set_title("Missing Values in Each Column")
    plt.xticks(rotation=45)

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Dataset Filters")

    selected_city = st.multiselect(
        "Select City",
        options=sorted(df["city"].unique()),
        default=sorted(df["city"].unique())
    )

    selected_status = st.multiselect(
        "Select Loan Approval Status",
        options=df["loan_approved"].unique().tolist(),
        default=df["loan_approved"].unique().tolist()
    )

    filtered_df = df[
        (df["city"].isin(selected_city)) &
        (df["loan_approved"].isin(selected_status))
    ]

    st.write("Filtered Records:", filtered_df.shape[0])

    if filtered_df.empty:

        st.warning(
            "No data available for the selected filters. "
            "Please select at least one city and approval status."
        )

    else:

        st.subheader("Filtered Dataset")

        st.dataframe(
            filtered_df,
            use_container_width=True
        )

        st.subheader("Loan Approval Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.countplot(
            data=filtered_df,
            x="loan_approved",
            ax=ax
        )

        ax.set_title("Loan Approval Distribution")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Income Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.histplot(
            data=filtered_df,
            x="income",
            kde=True,
            ax=ax
        )

        ax.set_title("Income Distribution")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Credit Score Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.histplot(
            data=filtered_df,
            x="credit_score",
            kde=True,
            ax=ax
        )

        ax.set_title("Credit Score Distribution")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Loan Amount Box Plot")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.boxplot(
            data=filtered_df,
            y="loan_amount",
            ax=ax
        )

        ax.set_title("Loan Amount Distribution")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Credit Score vs Loan Approval")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.boxplot(
            data=filtered_df,
            x="loan_approved",
            y="credit_score",
            ax=ax
        )

        ax.set_title("Credit Score vs Loan Approval")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Income vs Loan Approval")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.boxplot(
            data=filtered_df,
            x="loan_approved",
            y="income",
            ax=ax
        )

        ax.set_title("Income vs Loan Approval")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Correlation Heatmap")

        numeric_df = filtered_df.select_dtypes(
            include="number"
        ).drop(
            columns=["points"],
            errors="ignore"
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="RdYlGn",
            ax=ax
        )

        ax.set_title("Correlation Between Numerical Features")

        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Statistical Summary")

        st.dataframe(filtered_df.describe())


# MODEL COMPARISON
elif page == "Model Comparison":

    st.title("Model Comparison")

    st.subheader("Model Performance")

    st.dataframe(
        comparison,
        use_container_width=True
    )

    st.subheader("Accuracy Comparison")

    st.bar_chart(
        comparison.set_index("Model")[["Test Accuracy"]]
    )

    st.subheader("Best Model")

    st.success(
        "SVC achieved the highest test accuracy of 97.5%."
    )


# PREDICTION
elif page == "Prediction":

    st.title("Loan Approval Prediction")

    st.write(
        "Enter your details below to predict loan approval."
    )

    st.subheader("Select Machine Learning Model")

    selected_model = st.selectbox(
        "Choose a model:",
        list(models.keys())
    )

    st.subheader("Enter Loan Details")

    col1, col2 = st.columns(2)

    with col1:

        city = st.text_input(
            "City",
            placeholder="Enter your city"
        )

        income = st.number_input(
            "Income",
            min_value=0,
            step=1000
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=0
        )

    with col2:

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            step=1000
        )

        years_employed = st.number_input(
            "Years Employed",
            min_value=0
        )

    if st.button("Predict Loan Approval"):

        if not city.strip():

            st.warning("Please enter your city.")

        elif income <= 0 or credit_score <= 0 or loan_amount <= 0:

            st.warning("Please enter valid positive values.")

        else:

            input_data = pd.DataFrame({
                "city": [city.strip()],
                "income": [income],
                "credit_score": [credit_score],
                "loan_amount": [loan_amount],
                "years_employed": [years_employed]
            })

            model = models[selected_model]

            prediction = model.predict(input_data)

            st.subheader("Prediction Result")

            if prediction[0]:

                st.success(
                    "Your loan is predicted to be APPROVED."
                )

            else:

                st.warning(
                    "Your loan is predicted to be NOT APPROVED."
                )
