# Prompt Injection Detection Using Machine Learning

## Project Description
This project builds a reproducible Python workflow and a beginner-friendly machine learning classifier for identifying prompt injection attempts in text prompts. The project uses the Hugging Face dataset `deepset/prompt-injections` and demonstrates data ingestion, cleaning, exploratory analysis, visualization, text vectorization, classical model training, and prediction.

## What I Built
- A Jupyter Notebook named `data_workflow.ipynb`
- Modular Python scripts for training and inference
- A reproducible `requirements.txt` file
- A written summary report in PDF format
- Supporting folders for results, models, diagrams, and data notes

## Dataset Used
- **Dataset name:** deepset/prompt-injections
- **Link:** https://huggingface.co/datasets/deepset/prompt-injections

## How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Open and run the notebook
```bash
jupyter notebook
```
Then open `data_workflow.ipynb` and run all cells from top to bottom.

### 3. Optional: run the Python scripts
Train the model:
```bash
python train_model.py
```

Run the prediction demo:
```bash
python predict.py
```

Or use the simple CLI:
```bash
python main.py
```

## Bias Awareness
Poor data cleaning can introduce bias by removing informative records unevenly, preserving label imbalance, or allowing noisy text artifacts to influence the classifier more than the underlying meaning. In this project, cleaning is kept simple and transparent so the workflow is easy to audit and reproduce.

## Future Integration Reflections
### How would this workflow change for a larger ML project?
A larger ML project would add stronger validation, experiment tracking, hyperparameter tuning, model comparison, and automated testing for data drift and schema changes.

### How would you prepare this data for neural networks?
For neural networks, I would preserve more of the original text, tokenize it with a modern NLP tokenizer, manage sequence lengths, and use a train/validation/test split designed for model tuning.

### How could agentic automation help?
Agentic automation could monitor new prompts in real time, trigger the classifier automatically, route suspicious prompts for analyst review, and log evidence for a future AI security platform such as SentinelAI-RT.
