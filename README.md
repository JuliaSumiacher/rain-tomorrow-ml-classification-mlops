# Rain Tomorrow — Classification & MLOps Pipeline

A machine learning project focused on rain prediction (`weatherAUS.csv`), covering notebook experimentation, MLOps workflows, and containerized preprocessing and inference.

## Project Structure

- `rain-tomorrow-ml-classification-mlops.ipynb` — Jupyter notebook for classification modeling and analysis.
- `TP_clasificacion_MLOPs/mlops.ipynb` — Jupyter notebook covering MLOps steps and workflows.
- `TP_clasificacion_MLOPs/docker/Dockerfile` — Container image definition for running the inference pipeline.
- `TP_clasificacion_MLOPs/docker/preprocesador.py` — Data preprocessing module.
- `TP_clasificacion_MLOPs/docker/inference.py` — Inference script using `model.pkl`.
- `TP_clasificacion_MLOPs/docker/model.pkl` — Serialized trained classification model.
- `TP_clasificacion_MLOPs/docker/requirements.txt` — Python dependencies for the Docker environment.
- `TP_clasificacion_MLOPs/files/` — Dataset splits and sample input/output CSV files (`train.csv`, `test.csv`, `input.csv`, `output.csv`).
- `data/weatherAUS.csv` — Weather dataset used for rain classification.
- `data/house-prices-tp.csv` — Additional dataset included in `data/`.

## Usage

### Notebooks

1. Open `rain-tomorrow-ml-classification-mlops.ipynb` or `TP_clasificacion_MLOPs/mlops.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Run the cells sequentially to explore data, train/evaluate models, and follow the MLOps workflow.

### Docker Inference

Build and run the container from `TP_clasificacion_MLOPs/docker`:

```bash
docker build -t rain-tomorrow-inference .
docker run --rm rain-tomorrow-inference
