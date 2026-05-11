# Prompt Injection Detection Using Machine Learning

## Project Description
This project builds a reproducible Python workflow and machine learning classifier for identifying prompt injection attempts in text prompts. The project uses the Hugging Face dataset `deepset/prompt-injections` and demonstrates data ingestion, cleaning, exploratory analysis, visualization, text vectorization, model training, and prediction.

## What I Built
- `data_workflow.ipynb`
- Reproducible `requirements.txt`
- Machine learning workflow using TF-IDF and Logistic Regression
- Visualizations and exploratory analysis
- `module_summary.pdf`
- GitHub repository with version control and branching

## Dataset Used
- **Dataset:** deepset/prompt-injections
- **Link:** https://huggingface.co/datasets/deepset/prompt-injections

## How to Run the Project

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook
```bash
jupyter notebook
```

Open `data_workflow.ipynb` and run all cells from top to bottom.

## Reproducibility
This project includes:
- A reproducible Jupyter Notebook
- A `requirements.txt` file
- Version-controlled GitHub workflow with multiple commits and branches

## Bias Awareness
Poor data cleaning can introduce bias by removing or altering important records unevenly across classes. This project uses simple and transparent preprocessing steps to reduce that risk.

## Future Integration Reflections

### Larger ML Projects
Future projects could include automated validation, experiment tracking, and model comparison workflows.

### Neural Network Preparation
Neural network workflows would require tokenization, sequence management, and larger training datasets.

### Agentic Automation Potential
Agentic systems could automate prompt monitoring, threat detection, and analyst alerting in real time.
