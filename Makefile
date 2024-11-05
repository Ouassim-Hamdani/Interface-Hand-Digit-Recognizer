install:
	@pip install -r requirements.txt
run-main:
	@python src/main.py
run-app:
	@streamlit run src/app.py
docker:
	@docker compose up --build