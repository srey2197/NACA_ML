import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import warnings
from statistics import stdev, mean

warnings.filterwarnings(action="ignore", module="sklearn", message="^internal gelsd")

def main():

    data = pd.read_csv("data.csv")

    df = pd.DataFrame(data)

    X = df[['Frequency','AoA','Chord','Velocity','SuctionSideDisp']]
    y = df['SoundPressure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state = 50)

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    print('Intercept: {}'.format(regr.intercept_))
    print('Coefficients: {}'.format(regr.coef_))
    print("Mean squared error: %.5f" % mean_squared_error(y_test, y_pred))
    print('Variance score: %.5f' % r2_score(y_test, y_pred))

    newFrequency, newAoA, newChord, newVelocity, newSuctionSideDisp = 630, 4, 0.3048, 39.6, 0.00579636
    print ('Predicted Sound Pressure in dB: {}'.format(regr.predict([[newFrequency, newAoA, newChord, newVelocity, newSuctionSideDisp]])))

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print_model = model.summary()

    error = y_test - y_pred
    print("Average Error: {}".format(mean(error)))
    print("Standard Deviation: {}".format(stdev(error)))

    plt.hist(error, bins = 75, color = 'green')
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.show()

def csvWriter():

    with open('airfoil_self_noise.txt', 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split("\t") for line in stripped if line)

        with open('data.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('Frequency', 'AoA', 'Chord', 'Velocity', 'SuctionSideDisp', 'SoundPressure'))
            writer.writerows(lines)

def plot(df):

    plt.scatter(df['Frequency'], df['SoundPressure'], color='red')
    plt.title('SoundPressure Vs Frequency', fontsize=14)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('SoundPressure', fontsize=14)
    plt.grid(True)
    plt.show()

    plt.scatter(df['Chord'], df['SoundPressure'], color='black')
    plt.title('SoundPressure Vs Chord', fontsize=14)
    plt.xlabel('Chord', fontsize=14)
    plt.ylabel('SoundPressure', fontsize=14)
    plt.grid(True)
    plt.show()

def gradientDescent(x,y):

    m_curr = 0
    b_curr = 0
    iter = 1000
    n = len(x)
    learning_rate = 0.0000005

    for i in range(iter):
        y_pred = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y - y_pred)])
        md = -(2/n) * sum(x * (y - y_pred))
        bd = -(2/n) * sum(y - y_pred)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("iter {}: m {}, b {}, cost {}".format(i, m_curr, b_curr, cost))

main()
