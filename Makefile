deploy_app:
	docker build -t milk-quality-app:latest -f app/Dockerfile ./app
	docker run -d -p 7777:8501 \
		--name milk-quality-app-container \
		--network milk-app \
		-e SQL_USER=root \
		-e SQL_PASSWORD=root \
		-e DB_NAME=milk_quality_prediction \
		-e DB_PORT=5555 \
		milk-quality-app:latest

deploy_sql:
	docker build -t mysql-milk-quality:latest -f database/Dockerfile_sql ./database/
	docker run -d -p 5555:3306 --name mysql-container --network milk-app mysql-milk-quality:latest