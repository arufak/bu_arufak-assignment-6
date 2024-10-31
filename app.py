from flask import Flask, render_template, request, url_for
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64

app = Flask(__name__)

def generate_plots(N, mu, sigma2, S):
    # STEP 1
    # Generate random dataset X and Y with normal additive error
    X = np.random.uniform(0, 1, N)
    Y = X + np.random.normal(mu, np.sqrt(sigma2), N)
    
    # Reshape X for scikit-learn
    X_reshaped = X.reshape(-1, 1)
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_reshaped, Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # Generate scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, alpha=0.5, label='Data points')
    plt.plot(X, model.predict(X_reshaped), color='red', label='Regression line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Linear Regression: y = {slope:.2f}x + {intercept:.2f}')
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    # Step 2: Run S simulations
    slopes = []
    intercepts = []

    for _ in range(S):
        # Generate random X values
        X_sim = np.random.uniform(0, 1, N)
        X_sim_reshaped = X_sim.reshape(-1, 1)
        
        # Generate Y values with normal additive error
        Y_sim = X_sim + np.random.normal(mu, np.sqrt(sigma2), N)
        
        # Fit linear regression model
        sim_model = LinearRegression()
        sim_model.fit(X_sim_reshaped, Y_sim)
        
        # Store slopes and intercepts
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Plot histograms of slopes and intercepts
    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of more extreme slopes and intercepts
    slope_more_extreme = sum(s > slope for s in slopes) / S
    intercept_more_extreme = sum(i < intercept for i in intercepts) / S

    return plot1_path, plot2_path, slope_more_extreme, intercept_more_extreme

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get user input with validation
            N = max(10, min(1000, int(request.form["N"])))  # Limit between 10 and 1000
            mu = float(request.form["mu"])
            sigma2 = max(0.0001, float(request.form["sigma2"]))  # Ensure positive variance
            S = max(100, min(10000, int(request.form["S"])))  # Limit between 100 and 10000

            # Generate plots and results
            plot1, plot2, slope_extreme, intercept_extreme = generate_plots(N, mu, sigma2, S)

            return render_template("index.html", 
                                plot1=plot1, 
                                plot2=plot2,
                                slope_extreme=slope_extreme, 
                                intercept_extreme=intercept_extreme,
                                success=True)

        except Exception as e:
            return render_template("index.html", 
                                error=f"An error occurred: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)