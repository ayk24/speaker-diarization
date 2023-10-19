run:
	poetry run python main.py

lint:
	poetry run isort .
	poetry run black .
	poetry run mypy .