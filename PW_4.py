import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az
from sklearn.linear_model import LinearRegression
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="

def comparison_function():
    print("Генеруємо дані...")
    np.random.seed(42)
    n_samples = 50
    temperature = np.linspace(10, 35, n_samples)
    energy = 10 + 2.5 * temperature + np.random.normal(0, 7, size=n_samples)
    df = pd.DataFrame({'Temperature': temperature, 'Energy': energy})

    print("Класична модель (OLS)...")
    X = df[['Temperature']].values
    y = df['Energy'].values
    ols_model = LinearRegression()
    ols_model.fit(X, y)
    y_pred_ols = ols_model.predict(X)

    print("Байєсівське моделювання (MCMC)...")
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=20)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=10)

        mu = alpha + beta * temperature

        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=energy)

        trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True, cores=1)

    print(f"Класична модель (OLS):")
    print(f" - Перетин (Intercept): {ols_model.intercept_:.4f}")
    print(f" - Коефіцієнт нахилу (Beta): {ols_model.coef_[0]:.4f}")

    print("\nБайєсівське моделювання (MCMC):")
    summary = az.summary(trace, var_names=['alpha', 'beta', 'sigma'])
    print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])
    print("-" * 40)

    plt.figure(figsize=(12, 7))
    plt.scatter(temperature, energy, color='black', alpha=0.5, label='Фактичні дані')

    plt.plot(temperature, y_pred_ols, color='red', linestyle='--', linewidth=2,label=f'OLS лінія (Beta={ols_model.coef_[0]:.2f})')

    post = trace.posterior
    n_lines = 100
    chain_idx = np.random.randint(0, post.chain.size, size=n_lines)
    draw_idx = np.random.randint(0, post.draw.size, size=n_lines)

    for c, d in zip(chain_idx, draw_idx):
        line = post['alpha'][c, d].values + post['beta'][c, d].values * temperature
        plt.plot(temperature, line, color='skyblue', alpha=0.1)

    mean_alpha = post['alpha'].mean().values
    mean_beta = post['beta'].mean().values
    plt.plot(temperature, mean_alpha + mean_beta * temperature, color='blue', linewidth=2,
             label='Байєсівське середнє (MCMC)')

    plt.xlabel('Температура (°C)')
    plt.ylabel('Енергоспоживання (кВт/год)')
    plt.title('Порівняння моделей: Енергоспоживання vs Температура')
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.show()

if __name__ == '__main__':
    comparison_function()