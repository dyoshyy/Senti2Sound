run:
	@echo "Running app.py"
	gunicorn --bind :8000 --workers 2 src.app:app

dev:
	@echo "Running app.py"
	python src/app.py