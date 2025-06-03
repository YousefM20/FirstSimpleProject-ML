import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import   mean_squared_error, mean_absolute_error, r2_score

df= pd.read_csv ("Task 5 Dataset.csv")
X = df[['X']].values
y =df['Y'].values.reshape(-1,1)

#normalize 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#visualize 
plt.scatter(X_scaled, y , color='pink')
plt.title("Normalized Data")
plt.xlabel("X (normalized)")
plt.ylabel("Y")
plt.show()

def compute_mse(y_true, y_pred):
        return np.mean((y_true - y_pred) **2)


def gradint_descent(X ,y, lr=0.01,  epochs=1000):
    m = X.shape[0]
    X_b = np.c_[np.ones((m, 1)), X]  
    theta = np.zeros((2, 1)) 

    for epoch in range(epochs):
        gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
        theta -= lr * gradients
    return theta

theta= gradint_descent(X_scaled, y)

#Predict 
X_b = np.c_[np.ones((X_scaled.shape[0], 1)),X_scaled]
y_pred_custom = X_b.dot(theta)

#evaluation 
mse =compute_mse(y, y_pred_custom)
rmse =  np.sqrt(mse)
mae = mean_absolute_error(y, y_pred_custom)
r2 = r2_score(y, y_pred_custom)

print(" custom model perf.:")
print (f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R2: {r2:.4f}")


#scikit-learn
model = LinearRegression()
model.fit(X_scaled, y)
y_pred_sklearn = model.predict(X_scaled)

mse_sklearn = mean_squared_error(y, y_pred_sklearn)
rmse_sklearn = np.sqrt(mse_sklearn)
mae_sklearn = mean_absolute_error(y, y_pred_sklearn)
r2_sklearn = r2_score(y, y_pred_sklearn)

print("sklearn model per.:")
print(f"MSE: {mse_sklearn:.4f}, RMSE: {rmse_sklearn:.6f}, MAE: {mae_sklearn:.6f}, R2: {r2_sklearn:.4f}")

print(" custom theta (bias, weight):", theta.ravel())
print("sklearn Coefficients:", model.intercept_, model.coef_)

#piecewise 
def piecewise_linear_fit(X, y, breakpoints):
    segments = []
    for i in range(len(breakpoints) - 1):
        mask = (X >= breakpoints[i]) & (X < breakpoints[i+1])
        model = LinearRegression().fit(X[mask], y[mask])
        segments.append((breakpoints[i], breakpoints[i+1], model))
    return segments

#breakpoints 
breakpoints = [X.min(), X.mean(), X.max()]
segments = piecewise_linear_fit(X, y, breakpoints)  

#evaluate
plt.scatter(X, y, label='Data', color='gray')
for i, (start, end, model) in enumerate(segments):
    x_range = np.linspace(start, end, 100).reshape(-1, 1)
    y_range = model.predict(x_range)
    plt.plot(x_range, y_range, label=f'Segment {i+1}: {start:.2f} to {end:.2f}')
plt.title("Piecewise Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

