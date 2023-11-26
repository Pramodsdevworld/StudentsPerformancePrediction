import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from math import ceil
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame()

# Create a Streamlit app
st.title("Student Performance Analysis")
data = None
#Add a sidebar for navigation
st.sidebar.title("Navigation")
selected_tab = st.sidebar.radio(
    "Go to", ("Data Upload", "Data Preprocessing", "Data Analysis", "Prediction"))

# Data Upload tab
if selected_tab == "Data Upload":
    st.header("Data Upload")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        st.success("Data uploaded successfully!")
        st.write("Data Shape:")
        st.write(data.shape)
        st.write("Data Preview:")
        st.write(data.head())

# Data Preprocessing tab
elif selected_tab == "Data Preprocessing":
    data = st.session_state.data  # Retrieve data from the session state

    st.header("Data Preprocessing")
    if data is None:
        st.warning("Please upload data first in the 'Data Upload' tab.")
    else:
        # Retrieve data from the session state
        numeric_features = [
            feature for feature in data.columns if data[feature].dtype != "object"]
        categorical_features = [
            feature for feature in data.columns if data[feature].dtype == "object"]
        st.write("Numerical Features:")
        st.write(numeric_features)
        st.write("Categorical Features:")
        st.write(categorical_features)
        # Add preprocessing steps here
        data.isnull().sum()
        data.duplicated().sum()
        data.info()
        st.write(data.describe())
        data.select_dtypes('object').nunique()
        no_of_columns = data.shape[0]
        percentage_of_missing_data = data.isnull().sum()/no_of_columns
        st.write(percentage_of_missing_data)
        st.write("Categories in 'gender' variable:  ", end=" ")
        st.write(data["gender"].unique())

        st.write("Categories in 'race/ethnicity' variable:  ", end=" ")
        st.write(data["race/ethnicity"].unique())

        st.write("Categories in 'parental level of education' variable:  ", end=" ")
        st.write(data["parental level of education"].unique())

        st.write("Categories in 'lunch' variable:  ", end=" ")
        st.write(data["lunch"].unique())

        st.write("Categories in 'test preparation course' variable:  ", end=" ")
        st.write(data["test preparation course"].unique())

# Data Analysis tab
elif selected_tab == "Data Analysis":
    data = st.session_state.data  # Retrieve data from the session state

    st.header("Data Analysis")
    if data is None:
        st.warning("Please upload data first in the 'Data Upload' tab.")
    else:
        # Retrieve data from the session state
        # Add data analysis and visualization here
        data[['lunch', 'gender', 'math score', 'writing score',
              'reading score']].groupby(['lunch', 'gender']).agg('median')
        data[['test preparation course',
              'gender',
              'math score',
              'writing score',
              'reading score']].groupby(['test preparation course', 'gender']).agg('median')
        # Create a figure with two subplots
        f, ax = plt.subplots(1, 2, figsize=(8, 6))

        # Create a countplot of the 'gender' column and add labels to the bars
        sns.countplot(x=data['gender'], data=data,
                      palette='bright', ax=ax[0], saturation=0.95)
        for container in ax[0].containers:
            ax[0].bar_label(container, color='black', size=15)

        # Set font size of x-axis and y-axis labels and tick labels
        ax[0].set_xlabel('Gender', fontsize=14)
        ax[0].set_ylabel('Count', fontsize=14)
        ax[0].tick_params(labelsize=14)

        # Create a pie chart of the 'gender' column and add labels to the slices
        plt.pie(x=data['gender'].value_counts(), labels=['Male', 'Female'], explode=[
                0, 0.1], autopct='%1.1f%%', shadow=True, colors=['#ff4d4d', '#ff8000'], textprops={'fontsize': 14})

        # Display the plot
        st.pyplot(plt)
        # Define a color palette for the countplot
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        # blue, orange, green, red, purple are respectiively the color names for the color codes used above

        # Create a figure with two subplots
        f, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Create a countplot of the 'race/ethnicity' column and add labels to the bars
        sns.countplot(x=data['race/ethnicity'], data=data,
                      palette=colors, ax=ax[0], saturation=0.95)
        for container in ax[0].containers:
            ax[0].bar_label(container, color='black', size=14)

        # Set font size of x-axis and y-axis labels and tick labels
        ax[0].set_xlabel('Race/Ethnicity', fontsize=14)
        ax[0].set_ylabel('Count', fontsize=14)
        ax[0].tick_params(labelsize=14)

        # Create a dictionary that maps category names to colors in the color palette
        color_dict = dict(zip(data['race/ethnicity'].unique(), colors))
        # Map the colors to the pie chart slices
        pie_colors = [color_dict[race]
                      for race in data['race/ethnicity'].value_counts().index]

        # Create a pie chart of the 'race/ethnicity' column and add labels to the slices
        plt.pie(x=data['race/ethnicity'].value_counts(), labels=data['race/ethnicity'].value_counts().index,
                explode=[0.1, 0, 0, 0, 0], autopct='%1.1f%%', shadow=True, colors=pie_colors, textprops={'fontsize': 14})

        # Set the aspect ratio of the pie chart to 'equal' to make it a circle
        plt.axis('equal')

        # Display the plot
        st.pyplot(plt)
        plt.rcParams['figure.figsize'] = (15, 9)
        plt.style.use('fivethirtyeight')

        data['parental level of education'] = pd.Categorical(
            data['parental level of education'])
        plt.figure(figsize=(12, 6))
        sns.countplot(x='parental level of education',
                      data=data, palette='Blues')
        plt.title('Comparison of Parental Education',
                  fontweight=30, fontsize=20)
        plt.xlabel('Degree')
        plt.ylabel('count')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        plt.rcParams['figure.figsize'] = (20, 9)
        plt.style.use('tableau-colorblind10')

        # Define the range of values you want to display on the x-axis
        filtered_data = data[(data['math score'] >= 0) &
                             (data['math score'] <= 100)]
        ax = sns.countplot(x=filtered_data['math score'], palette='BuPu')

        ax.set_facecolor('white')

        plt.title('Comparison of math scores', fontweight=30, fontsize=20)
        plt.xlabel('score')
        plt.ylabel('count')
        plt.xticks(rotation=45, fontsize=12)  # Increase x-axis label font size
        plt.yticks(fontsize=12)  # Increase y-axis label font size

        st.pyplot(plt)
        import warnings
        warnings.filterwarnings('ignore')

        data['total_score'] = data['math score'] + \
            data['reading score'] + data['writing score']
        plt.figure(figsize=(12, 6))
        sns.distplot(data['total_score'], color='magenta')

        plt.title('comparison of total score of all the students',
                  fontweight=30, fontsize=20)
        plt.xlabel('total score scored by the students')
        plt.ylabel('count')
        st.pyplot(plt)
        from math import *
        import warnings
        warnings.filterwarnings('ignore')

        data['percentage'] = data['total_score']/3

        for i in range(0, 1000):
            data['percentage'][i] = ceil(data['percentage'][i])

        plt.rcParams['figure.figsize'] = (15, 9)
        plt.figure(figsize=(12, 6))
        sns.distplot(data['percentage'], color='orange')

        plt.title('Comparison of percentage scored by all the students',
                  fontweight=30, fontsize=20)
        plt.xlabel('Percentage scored')
        plt.ylabel('Count')
        st.pyplot(plt)
        # Set figure size
        plt.rcParams['figure.figsize'] = (12, 9)

        # First row of pie charts
        plt.subplot(2, 3, 1)
        size = data['gender'].value_counts()
        labels = 'Female', 'Male'
        color = ['red', 'green']
        plt.pie(size, colors=color, labels=labels, autopct='%.2f%%')
        plt.title('Gender', fontsize=20)
        plt.axis('off')

        plt.subplot(2, 3, 2)
        size = data['race/ethnicity'].value_counts()
        labels = 'Group C', 'Group D', 'Group B', 'Group E', 'Group A'
        color = ['red', 'green', 'blue', 'cyan', 'orange']
        plt.pie(size, colors=color, labels=labels, autopct='%.2f%%')
        plt.title('Race/Ethnicity', fontsize=20)
        plt.axis('off')

        plt.subplot(2, 3, 3)
        size = data['lunch'].value_counts()
        labels = 'Standard', 'Free'
        color = ['red', 'green']
        plt.pie(size, colors=color, labels=labels, autopct='%.2f%%')
        plt.title('Lunch', fontsize=20)
        plt.axis('off')
        st.pyplot(plt)
        # Second row of pie charts
        plt.subplot(2, 3, 4)
        size = data['test preparation course'].value_counts()
        labels = 'None', 'Completed'
        color = ['red', 'green']
        plt.pie(size, colors=color, labels=labels, autopct='%.2f%%')
        plt.title('Test Course', fontsize=20)
        plt.axis('off')

        plt.subplot(2, 3, 5)
        size = data['parental level of education'].value_counts()
        labels = 'Some College', "Associate's Degree", 'High School', 'Some High School', "Bachelor's Degree", "Master's Degree"
        color = ['red', 'green', 'blue', 'cyan', 'orange', 'grey']
        plt.pie(size, colors=color, labels=labels, autopct='%.2f%%')
        plt.title('Parental Education', fontsize=20)
        plt.axis('off')
        st.pyplot(plt)

        def getgrade(percentage):
            if (percentage >= 90):
                return 'O'
            if (percentage >= 80):
                return 'A'
            if (percentage >= 70):
                return 'B'
            if (percentage >= 60):
                return 'C'
            if (percentage >= 40):
                return 'D'
            else:
                return 'E'

        data['grades'] = data.apply(
            lambda x: getgrade(x['percentage']), axis=1)

        st.write(data['grades'].value_counts())
# Prediction tab
elif selected_tab == "Prediction":
    data = st.session_state.data  # Retrieve data from the session state

    st.header("Prediction")
    if data is None:
        st.warning("Please upload data first in the 'Data Upload' tab.")
    else:
        # Retrieve data from the session state
        # Add prediction and model training here
        X = data.drop(columns="math score", axis=1)
        y = data["math score"]
        num_features = X.select_dtypes(exclude="object").columns
        cat_features = X.select_dtypes(include="object").columns

        numeric_transformer = StandardScaler()
        oh_transformer = OneHotEncoder()
        from sklearn.preprocessing import LabelEncoder
        data = pd.get_dummies(
            data, columns=['gender', 'race/ethnicity', 'lunch', 'test preparation course'])
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Apply label encoding to each categorical column
        X['parental level of education'] = label_encoder.fit_transform(
            X['parental level of education'])

        preprocessor = ColumnTransformer([("OneHotEncoder", oh_transformer, cat_features), (
            "StandardScaler", numeric_transformer, num_features), ])
        X = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_train.shape, X_test.shape

        def evaluate_model(true, predicted):
            mae = mean_absolute_error(true, predicted)
            mse = mean_squared_error(true, predicted)
            rmse = np.sqrt(mean_squared_error(true, predicted))
            r2_square = r2_score(true, predicted)
            return mae, mse, rmse, r2_square
        models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor()
        }
        model_list = []
        r2_list = []

        for i in range(len(list(models))):

            model = list(models.values())[i]
            model.fit(X_train, y_train)  # Train model

        # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        # Evaluate Train and Test dataset
            model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(
                y_train, y_train_pred)

            model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(
                y_test, y_test_pred)

            st.write(list(models.keys())[i])
            model_list.append(list(models.keys())[i])

            st.write('Model performance for Training set')
            st.write(
                "- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
            st.write("- Mean Squared Error: {:.4f}".format(model_train_mse))
            st.write("- Mean Absolute Error: {:.4f}".format(model_train_mae))
            st.write("- R2 Score: {:.4f}".format(model_train_r2))

            st.write('----------------------------------')
            st.write('Model performance for Test set')
            st.write(
                "- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
            st.write("- Mean Squared Error: {:.4f}".format(model_test_rmse))
            st.write("- Mean Absolute Error: {:.4f}".format(model_test_mae))
            st.write("- R2 Score: {:.4f}".format(model_test_r2))
            r2_list.append(model_test_r2)
            st.write('='*35)
            st.write('\n')
        # Define hyperparameter ranges for each model
        param_grid = {
            "Linear Regression": {},
            "Lasso": {"alpha": [1]},
            "K-Neighbors Regressor": {"n_neighbors": [3, 5, 7], },
            "Decision Tree": {"max_depth": [3, 5, 7], 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
            "Random Forest Regressor": {'n_estimators': [8, 16, 32, 64, 128, 256], "max_depth": [3, 5, 7]},
            "Gradient Boosting": {'learning_rate': [.1, .01, .05, .001], 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                                  'n_estimators': [8, 16, 32, 64, 128, 256]},
            "AdaBoost Regressor": {'learning_rate': [.1, .01, 0.5, .001], 'n_estimators': [8, 16, 32, 64, 128, 256]}
        }

        model_list = []
        r2_list = []

        for model_name, model in models.items():
            # Create a scorer object to use in grid search
            scorer = make_scorer(r2_score)
        # Perform grid search to find the best hyperparameters
            grid_search = GridSearchCV(
                model,
                param_grid[model_name],
                scoring=scorer,
                cv=5,
                n_jobs=-1
            )

        # Train the model with the best hyperparameters
            grid_search.fit(X_train, y_train)
        # Make predictions
            y_train_pred = grid_search.predict(X_train)
            y_test_pred = grid_search.predict(X_test)
        # Evaluate Train and Test dataset
            model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(
                y_train, y_train_pred)
            model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(
                y_test, y_test_pred)
            st.write(model_name)
            model_list.append(model_name)
            st.write('Best hyperparameters:', grid_search.best_params_)
            st.write('Model performance for Training set')
            st.write(
                "- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
            st.write("- Mean Squared Error: {:.4f}".format(model_train_mse))
            st.write("- Mean Absolute Error: {:.4f}".format(model_train_mae))
            st.write("- R2 Score: {:.4f}".format(model_train_r2))
            st.write('----------------------------------')
            st.write('Model performance for Test set')
            st.write(
                "- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
            st.write("- Mean Squared Error: {:.4f}".format(model_test_rmse))
            st.write("- Mean Absolute Error: {:.4f}".format(model_test_mae))
            st.write("- R2 Score: {:.4f}".format(model_test_r2))
            r2_list.append(model_test_r2)
            st.write('='*35)
            st.write('\n')

        pd.DataFrame(list(zip(model_list, r2_list)), columns=[
                     'Model Name', 'R2_Score']).sort_values(by=["R2_Score"], ascending=False)
        plt.scatter(y_test, y_test_pred)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        st.pyplot(plt)
        sns.regplot(x=y_test, y=y_test_pred, ci=None, color='red')
        pred_df = pd.DataFrame(
            {'Actual Value': y_test, 'Predicted Value': y_test_pred, 'Difference': y_test-y_test_pred})
        pred_df
# Conclusion tab
# elif selected_tab == "Conclusion":
    # st.header("Conclusion")
    # data = st.session_state.data  # Retrieve data from the session state

    # if data is None:
    #     st.warning("Please upload data first in the 'Data Upload' tab.")
    # else:

    #     # Add your conclusions here
    #     key_takeaways = [
    #         "Identification of student performance prediction is important for many institutions.",
    #         "Linear regression gives better accuracy compared to other regression problems.",
    #         "Linear regression is the best fit for the problem.",
    #         "Linear regression provides an accuracy of 88%, giving out the most accurate results."
    #     ]

    #     # Display the conclusion section
    #     st.write("Conclusion:")
    #     st.write("------------")
    #     st.write("First, we started by defining our problem statement and exploring the algorithms suitable for student performance prediction.")
    #     st.write("We practically implemented various regression algorithms, including Linear Regression, Lasso, K-Neighbors Regressor, Decision Tree, Random Forest Regressor, and AdaBoost Regressor.")
    #     st.write(
    #         "We compared the performances of these models and found that Linear Regression outperforms the others.")
    #     st.write("Our Linear Regression model achieved an accuracy of 88%, making it the most accurate model for this prediction task.")
    #     st.write("\nKey Takeaways:")
    #     for i, takeaway in enumerate(key_takeaways, start=1):
    #         st.write(f"{i}. {takeaway}")

# Display the app
if "data" in st.session_state:
    st.session_state.data = data  # Store the data in session state
