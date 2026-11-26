.PHONY: install train evaluate predict test docker-build docker-run

install:
	pip install -e ".[dev]"

train:
	python -m rain_tomorrow.train

evaluate:
	python -m rain_tomorrow.evaluate

predict:
	python -m rain_tomorrow.inference

test:
	pytest

docker-build:
	docker build -t rain-tomorrow-inference .

docker-run:
	docker run --rm -v "$(CURDIR)/data/sample:/data" rain-tomorrow-inference
