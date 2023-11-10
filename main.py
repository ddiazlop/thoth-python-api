from fastapi import FastAPI
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from models.expensesByMonth import ExpensesByMonth
import numpy as np
from sklearn.linear_model import LinearRegression

from models.prediction import Prediction

app = FastAPI()


@app.post('/predict/expensesPerMonth/linear', response_model=Prediction)
async def predict_expenses_per_month(expenses_per_month : list[ExpensesByMonth]) -> Prediction:
    # Convert the input data to a DataFrame
    df = pd.DataFrame([(expense.month, expense.amount) for expense in expenses_per_month], columns=['month', 'amount'])

    # Extract year and month from the 'month' column
    df['year'] = pd.to_datetime(df['month']).dt.year
    df['month'] = pd.to_datetime(df['month']).dt.month

    # Prepare the features (year and month) and the target variable (amount)
    X = df[['year', 'month']]
    y = df['amount']

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the last date in the dataset
    last_month_date = pd.to_datetime(str(df['year'].max()) + "-" +str(df['month'].max()), format='%Y-%m')

    # Predict expenses for the next month
    next_month_date = last_month_date + pd.DateOffset(months=1)
    next_month = np.array([[next_month_date.year, next_month_date.month]])

    # Provide column names for the prediction
    next_month_df = pd.DataFrame(next_month, columns=['year', 'month'])
    prediction = model.predict(next_month_df)[0]
    score = model.score(X, y)

    # Return the prediction
    return Prediction(prediction=prediction, confidence=score)

