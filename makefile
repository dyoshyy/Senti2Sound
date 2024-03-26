run:
	@echo "Running app.py"
	gunicorn src.app:app

dev:
	@echo "Running app.py"
	python src/app.py