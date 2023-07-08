run:
	poetry run python main.py

format:
	poetry run ruff --fix .
	poetry run black .
	poetry run isort --profile black . --gitignore