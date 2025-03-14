# ML Operations Project

This project demonstrates a machine learning pipeline for training, monitoring, and deploying models. It uses the following tools:

- **Flask**: For serving the model via a REST API.
- **MLflow**: For tracking experiments, logging metrics, and managing model versions.
- **Docker**: For containerizing the application and its dependencies.
- **Elasticsearch**: For storing and querying model performance metrics.
- **Kibana**: For visualizing the metrics stored in Elasticsearch.
- **Swagger**: For API documentation and testing.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Tools and Technologies](#tools-and-technologies)
3. [Setup and Installation](#setup-and-installation)
4. [Running the Project](#running-the-project)
5. [Accessing Services](#accessing-services)
6. [API Documentation](#api-documentation)
7. [Monitoring and Visualization](#monitoring-and-visualization)
8. [Docker Commands](#docker-commands)

---

## Project Overview

This project includes:
- A machine learning pipeline for preprocessing, training, and evaluating models.
- Integration with MLflow for experiment tracking and model management.
- A Flask-based API for serving predictions.
- Elasticsearch and Kibana for monitoring model performance.
- Docker for easy deployment and scalability.

---

## Tools and Technologies

### 1. **Flask**
- **Description**: A lightweight web framework for serving the trained model via a REST API.
- **Port**: `5000` (default for Flask).

### 2. **MLflow**
- **Description**: A platform for managing the machine learning lifecycle, including experiment tracking, model versioning, and deployment.
- **Port**: `5006` (MLflow UI).

### 3. **Docker**
- **Description**: A containerization platform for packaging the application and its dependencies.
- **Ports**: 
  - Flask API: `8080` (mapped to `5000` inside the container).
  - MLflow: `5006`.

### 4. **Elasticsearch**
- **Description**: A distributed search and analytics engine for storing and querying model performance metrics.
- **Port**: `9200` (HTTP), `9300` (TCP).

### 5. **Kibana**
- **Description**: A visualization tool for exploring and analyzing data stored in Elasticsearch.
- **Port**: `5601`.

### 6. **Swagger**
- **Description**: A tool for documenting and testing REST APIs.
- **Port**: Integrated with Flask API (`5000` or `8080` when using Docker).

---

## Setup and Installation

### Prerequisites
- Python 3.8+
- Docker
- Docker Compose

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MLoperations.git
   cd MLoperations
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt 
3. Start the docker containers:
    ```bash
    docker-compose up -d

Use the make file for reference
Accessing Services
Service	URL	Port
Flask API	http://localhost:8080	8080
MLflow UI	http://localhost:5006	5006
Elasticsearch	http://localhost:9200	9200
Kibana	http://localhost:5601	5601
Swagger UI	http://localhost:8080/docs	8080
