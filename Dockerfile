FROM python:3.12.7-slim
WORKDIR /app

# Install system dependencies if any are needed for python packages
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Install python dependencies matching the ML environment
RUN pip install mlflow==2.19.0 scikit-learn==1.4.2 pandas==2.2.2 numpy==1.26.4 matplotlib==3.8.4 seaborn==0.13.2

# Copy the trained model
COPY served_model /app/model

EXPOSE 5000

# Serve the model using MLflow with local env manager (since dependencies are already installed)
CMD ["mlflow", "models", "serve", "-m", "/app/model", "-h", "0.0.0.0", "-p", "5000", "--env-manager=local"]
