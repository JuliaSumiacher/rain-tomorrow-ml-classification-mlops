FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY models/model.pkl ./models/model.pkl

# Mount a directory containing input.csv at /data; predictions are written to /data/output.csv
ENTRYPOINT ["python", "-m", "rain_tomorrow.inference"]
CMD ["--input", "/data/input.csv", "--output", "/data/output.csv", "--model", "models/model.pkl"]
