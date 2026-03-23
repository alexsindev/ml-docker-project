# Iris Species Classifier

**Student ID:** st126112

A machine learning web application that classifies iris flowers by species using a Random Forest classifier, deployed with Flask and Docker.

## Project Overview

This project demonstrates a complete ML deployment pipeline:
- Training a Random Forest classification model on the Iris dataset
- Serving predictions via a Flask API
- Containerizing with Docker for deployment

## Tech Stack

- Python 3.9
- scikit-learn - Random Forest Classifier
- Flask - Web API
- Docker - Containerization
- Iris Dataset (built-in to scikit-learn)

## How to Run Locally

### Clone the repository
```bash
git clone https://github.com/alexsindev/ml-docker-project.git
cd ml-docker-project
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python train.py
```

### Run the Flask app
```bash
python app.py
```

### Visit in browser
```
http://localhost:5001
```

## Docker Deployment

### Build the Docker image
```bash
docker build -t iris-classifier:st126112 .
```

### Run the container
```bash
docker run -d -p 6112:5001 --name iris-app iris-classifier:st126112
```

### Verify the container is running
```bash
docker ps
```

### Stop and remove the container
```bash
docker stop iris-app
docker rm iris-app
```

## Model Details

| Item | Detail |
|---|---|
| Dataset | Iris (scikit-learn) |
| Algorithm | Random Forest Classifier |
| Features | Sepal length, sepal width, petal length, petal width |
| Hyperparameters | n_estimators=100, max_depth=5 |
| Accuracy | ~95% on validation set |

## API Endpoints

### Health Check
```bash
curl http://localhost:5001/health
```

### Prediction
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

## Author

- **Student ID:** st126112
- **Project:** ML Model Deployment with Docker
- **Course:** Machine Learning
# ml-docker-project
