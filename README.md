# 🌌 Atlas MLOps Capstone Project: End-to-End Text Sentiment Analysis & Deployment Pipeline

This repository implements a production-grade, end-to-end Machine Learning Operations (MLOps) pipeline for text sentiment analysis. The project demonstrates how to orchestrate a dataset's lifecycle from remote ingestion, data cleaning, feature engineering, model training, evaluation, model registry, containerization, deployment, CI/CD, and monitoring.

---

## 🎯 Purpose of the Project
In a real-world enterprise setting, deploying a machine learning model involves far more than just training a model in a Jupyter Notebook. The purpose of this capstone project is to design and implement a complete MLOps lifecycle to address the following challenges:
- **Reproducibility**: Version-control data and code dependencies together so any pipeline execution can be reproduced exactly.
- **Model Tracking & Registry**: Trace model parameters, metrics, and artifact versions during training, and manage the model lifecycle stages (Staging, Production) in a central registry.
- **Continuous Integration & Delivery (CI/CD)**: Automate linting, testing, and containerization using Docker, and automatically deploy the application to a production Kubernetes cluster.
- **Scalability**: Run the inference application inside a Kubernetes cluster (EKS) managed with auto-scaling worker nodes.
- **Monitoring & Observability**: Scraping application/model performance metrics at runtime and visualizing them to detect drift, service failure, and latency spikes.

---

## 🏛️ Architecture & System Design Flow

Below is the complete architectural layout of the pipeline, from S3 data ingestion to model inference monitoring:

```mermaid
flowchart TD
    subgraph Data Pipeline [Data & Preprocessing Pipeline (DVC)]
        A[AWS S3 Bucket / SSMS] -->|Ingest| B(data_ingestion.py)
        B -->|Raw Split| C[data/raw/train.csv & test.csv]
        C --> D(data_preprocessing.py)
        D -->|Interim Cleaned| E[data/interim/train_processed.csv & test_processed.csv]
        E --> F(feature_engineering.py)
        F -->|Processed BoW Features| G[data/processed/train_bow.csv & test_bow.csv]
        F -->|Save Vectorizer| H[models/vectorizer.pkl]
    end

    subgraph Training Pipeline [Model Training & Tracking]
        G --> I(model_building.py)
        I -->|Trained Model| J[models/model.pkl]
        J --> K(model_evaluation.py)
        G -->|Test Features| K
        K -->|Log Metrics/Params| L[MLflow Tracking Server on DagsHub]
        K -->|Save Metrics| M[reports/metrics.json]
        K -->|Save Run Info| N[reports/experiment_info.json]
        N --> O(register_model.py)
        O -->|Register & Transition to Staging| P[MLflow Model Registry on DagsHub]
    end

    subgraph CI/CD [CI/CD & Containerization]
        Q[GitHub Code Push] -->|Triggers| R[GitHub Actions CI/CD]
        R -->|Build & Package| S[Docker Image]
        S -->|Push| T[AWS ECR]
    end

    subgraph Production Deployment [Production Cluster]
        U[AWS EKS Cluster] -->|Pull Image| T
        U -->|Deploy Flask App Pods| V[Kubernetes Pods]
        W[LoadBalancer Service] -->|Route Traffic| V
        X[End Users] -->|HTTP Requests| W
    end

    subgraph Monitoring [Monitoring & Observability]
        Y[Prometheus EC2 Instance] -->|Scrape Metrics| W
        Z[Grafana EC2 Instance] -->|Visualize Metrics| Y
        Z -->|Dashboards| AA[MLOps Engineers / Operators]
    end
```

---

## 🛠️ Tools & Technologies Used (And Why)

Here is a detailed breakdown of the tools used in this project and the rationale behind their selection:

| Component | Tool / Technology | Role in Project | Why We Used It (Rationale) |
| :--- | :--- | :--- | :--- |
| **Virtual Environment** | Conda (`atlas`) | Environment isolation | Ensures a clean, reproducible workspace with Python 3.10, preventing version conflicts with local environments. |
| **Project Skeleton** | Cookiecutter Data Science | Template-driven structure | Standardizes repository layout, separating source code (`src`), models (`models`), reports (`reports`), and notebooks (`notebooks`). |
| **Data Storage / Ingestion** | AWS S3 / SSMS (SQL Server) | Raw data storage | AWS S3 serves as our robust, scalable remote object store for dataset ingestion. `s3_connection.py` loads `data.csv` directly into the ingestion phase. `ssms_connection.py` provides relational database compatibility. |
| **NLP Preprocessing** | NLTK (Natural Language Toolkit) | Text cleaning & Lemmatization | Used to clean raw tweets/reviews by removing HTML URLs, punctuation, digits, and stopwords, then lemmatizing words to their base form. |
| **Data & Pipeline Versioning** | DVC (Data Version Control) | Pipeline orchestration & data tracking | DVC links code commits with data hashes (via `.dvc` and `dvc.lock`), avoiding saving large datasets directly in Git. It tracks execution status through `dvc.yaml`. |
| **Feature Engineering** | Scikit-learn (`CountVectorizer`) | Text vectorization | Converts cleaned reviews into numerical Bag-of-Words (BoW) representation. |
| **Modeling & Classification** | Scikit-learn (`LogisticRegression`) | Classifier training | Fits a L1-regularized Logistic Regression classifier for high-speed, interpretable sentiment predictions (1 for positive, 0 for negative). |
| **Experiment Tracking** | MLflow | Parameter, metric, and artifact logging | Logs hyperparameter values (regularization strength, penalty) and metrics (Accuracy, Precision, Recall, AUC) during training. |
| **Registry & Collaboration** | DagsHub | Remote tracking & Git remote hosting | Hosts the repository and provides a managed, remote MLflow server & DagsHub Model Registry to transition models to "Staging" or "Production" stages. |
| **Web Service API** | Flask | Inference web application | Provides an HTTP API endpoint to accept review inputs, vectorize them using the saved `vectorizer.pkl`, and output predicted sentiments. |
| **Containerization** | Docker | Packaging web app and dependencies | Packages the Flask application into a lightweight, portable container image (`capstone-app`), guaranteeing identical behavior locally and in EKS. |
| **Orchestration / Cloud Host** | AWS EKS (Elastic Kubernetes Service) | Production model serving | Deploys Flask container pods onto Kubernetes. Out-of-the-box support for auto-scaling, high availability, and LoadBalancer services. |
| **Cluster Management** | `eksctl` & `kubectl` | Cluster creation & pod management | `eksctl` automates Kubernetes cluster setup on AWS using CloudFormation templates. `kubectl` allows resource management inside the cluster. |
| **Infrastructure-as-Code** | AWS CloudFormation | AWS resource orchestration | Behind the scenes, `eksctl` provisions EKS control planes and node groups using CloudFormation Stacks to ensure predictable deployments. |
| **CI / CD Pipeline** | GitHub Actions | Automated build and deploy | Automates linting, builds Docker containers, pushes them to AWS ECR, and triggers EKS cluster rollouts. |
| **Monitoring** | Prometheus | Real-time metric scraping | Scrapes performance metrics from the Flask web app running inside Kubernetes pods. |
| **Visualization** | Grafana | Metrics dashboarding | Visualizes metrics scraped by Prometheus, displaying real-time graphs for traffic, error rates, and system performance. |

---

## 📁 Repository Directory Structure

```text
├── .dvc/                   # DVC configuration and cache definitions
├── data/
│   ├── raw/                # Ingested train.csv and test.csv (tracked via DVC)
│   ├── interim/            # Cleaned train_processed.csv and test_processed.csv (tracked via DVC)
│   └── processed/          # Vectorized features train_bow.csv and test_bow.csv (tracked via DVC)
├── docs/                   # Sphinx documentation source code
├── models/
│   ├── vectorizer.pkl      # Trained CountVectorizer
│   └── model.pkl           # Trained Logistic Regression classifier
├── notebooks/              # Jupyter notebooks and exploratory scripts
│   ├── exp1.ipynb          # EDA & initial experiments
│   ├── exp2_bow_vs_tfidf.py# Feature extraction comparisons
│   └── exp3_lor_bow_hp.py  # Hyperparameter tuning script
├── reports/
│   ├── metrics.json        # Accuracy, precision, recall, and AUC metrics
│   └── experiment_info.json# Run ID and path configuration for model registration
├── src/                    # Source code for the project
│   ├── connections/        # S3 and SQL Server connections
│   │   ├── s3_connection.py
│   │   └── ssms_connection.py
│   ├── data/               # Ingestion and preprocessing pipelines
│   │   ├── data_ingestion.py
│   │   └── data_preprocessing.py
│   ├── features/           # Feature extraction modules
│   │   └── feature_engineering.py
│   ├── logger/             # Custom logging configurations
│   │   └── __init__.py
│   └── model/              # Model training, evaluation, and registration
│       ├── model_building.py
│       ├── model_evaluation.py
│       └── register_model.py
├── dvc.yaml                # DVC pipeline stages definition
├── params.yaml             # DVC pipeline configuration parameters
├── requirements.txt        # Python package dependencies
├── setup.py                # Setup script for packaging
└── projectflow.txt         # Step-by-step setup documentation
```

---

## ⚙️ How to Set Up and Run the Project

### 1. Environment Initialization
Create a dedicated environment with Python 3.10:
```bash
# Create and activate environment
conda create -n atlas python=3.10 -y
conda activate atlas

# Install project dependencies
pip install -r requirements.txt
```

### 2. Configure Remote Storage & DVC
We use AWS S3 as our remote data storage and DVC to manage data files without committing large assets to Git.
```bash
# Initialize DVC
dvc init

# (Optional Local Workspace setup)
mkdir local_s3
dvc remote add -d mylocal local_s3

# (Production S3 Workspace setup)
# Configure AWS CLI Credentials
aws configure

# Connect S3 Remote bucket
dvc remote add -d myremote s3://your-dvc-bucket-name
```

### 3. Pipeline Execution (Reproducing Experiments)
The training workflow is structured as a DVC pipeline. Modify configuration settings in `params.yaml`, then run the entire workflow:
```bash
# Execute DVC stages
dvc repro
```
DVC will verify data hashes, determine which steps changed, and run the necessary stages:
1. **Data Ingestion**: Downloads data from S3, splits train/test -> [data_ingestion.py](src/data/data_ingestion.py)
2. **Preprocessing**: Normalizes text (lowercase, removes stopwords, punctuation, lemmatizes words) -> [data_preprocessing.py](src/data/data_preprocessing.py)
3. **Feature Engineering**: Vectorizes text using Bag of Words -> [feature_engineering.py](src/features/feature_engineering.py)
4. **Model Building**: Trains a Logistic Regression model -> [model_building.py](src/model/model_building.py)
5. **Model Evaluation**: Logs metrics and saves reports -> [model_evaluation.py](src/model/model_evaluation.py)
6. **Model Registration**: Registers model to MLflow registry -> [register_model.py](src/model/register_model.py)

---

## 📊 Experiment Tracking & Model Registry with MLflow & DagsHub

All hyperparameters, evaluation metrics (accuracy, precision, recall, AUC), and model versions are tracked automatically inside MLflow, backed by DagsHub.

* **Tracking URI**: Track runs remotely on Dagshub using `mlflow.set_tracking_uri`.
* **Model Registry**: Once metrics are computed, [register_model.py](src/model/register_model.py) automatically pushes the candidate model to the DagsHub Model Registry and transitions it to the **Staging** stage for production testing.

To track models during local runs, authenticate with your DagsHub token:
```bash
set CAPSTONE_TEST=your_dagshub_auth_token
```

---

## 🐳 Containerization and Local Inference

To serve the sentiment analysis model, we deploy a Flask API.

1. **Build the Docker Image**:
   ```bash
   docker build -t capstone-app:latest .
   ```
2. **Run Container Locally**:
   ```bash
   docker run -p 8888:5000 -e CAPSTONE_TEST=your_dagshub_auth_token capstone-app:latest
   ```
The application will run locally at `http://localhost:8888`.

---

## ☸️ EKS Kubernetes Cluster Deployment

Production deployment is managed on Amazon Web Services (AWS) using Elastic Kubernetes Service (EKS).

### 1. Cluster Provisioning
Ensure `aws`, `kubectl`, and `eksctl` are installed and added to the System Path. Create the EKS cluster:
```bash
eksctl create cluster --name flask-app-cluster --region us-east-1 --nodegroup-name flask-app-nodes --node-type t3.small --nodes 1 --nodes-min 1 --nodes-max 1 --managed
```
*Behind the scenes, this generates CloudFormation templates to provision the EKS control plane and worker node auto-scaling groups.*

### 2. Configure kubectl
Point your local `kubectl` to the newly created EKS cluster:
```bash
aws eks --region us-east-1 update-kubeconfig --name flask-app-cluster
```

### 3. Deploy Application via Kubernetes manifests
Verify connection and deploy:
```bash
# Check connectivity
kubectl get nodes

# Deploy your deployment.yaml and service.yaml configurations
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Fetch the external IP of the LoadBalancer
kubectl get svc flask-app-service
```

---

## 📈 Monitoring & Observability with Prometheus & Grafana

To monitor model endpoints, response latency, and system memory, we deploy Prometheus and Grafana onto separate Ubuntu EC2 instances.

### 1. Prometheus Server Setup (Ubuntu EC2)
* Allow inbound access on port `9090` and SSH port `22`.
* Download and extract Prometheus:
  ```bash
  wget https://github.com/prometheus/prometheus/releases/download/v2.46.0/prometheus-2.46.0.linux-amd64.tar.gz
  tar -xvzf prometheus-2.46.0.linux-amd64.tar.gz
  sudo mv prometheus-2.46.0.linux-amd64 /etc/prometheus
  ```
* Configure the scrape targets in `/etc/prometheus/prometheus.yml`:
  ```yaml
  global:
    scrape_interval: 15s

  scrape_configs:
    - job_name: "flask-app"
      static_configs:
        - targets: ["<YOUR_LOADBALANCER_EXTERNAL_IP>:5000"]
  ```
* Start the Prometheus Server:
  ```bash
  /usr/local/bin/prometheus --config.file=/etc/prometheus/prometheus.yml
  ```

### 2. Grafana Dashboard Server Setup (Ubuntu EC2)
* Allow inbound access on port `3000` (Grafana Web UI) and SSH port `22`.
* Install Grafana on Ubuntu:
  ```bash
  wget https://dl.grafana.com/oss/release/grafana_10.1.5_amd64.deb
  sudo apt install ./grafana_10.1.5_amd64.deb -y
  sudo systemctl start grafana-server
  sudo systemctl enable grafana-server
  ```
* Access Grafana at `http://<ec2-public-ip>:3000`. Add your Prometheus server (`http://<prometheus-ec2-ip>:9090`) as a data source to begin charting endpoint metrics.

---

## 🧼 Resource Cleanup
To prevent ongoing AWS billing, delete all resources once your work is complete:
```bash
# Delete Kubernetes resources
kubectl delete deployment flask-app
kubectl delete service flask-app-service
kubectl delete secret capstone-secret

# Terminate EKS Cluster
eksctl delete cluster --name flask-app-cluster --region us-east-1

# Optional: delete ECR repositories and S3 buckets
```