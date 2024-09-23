# Use the official MySQL image from the Docker Hub
FROM mysql:8.0

# Set environment variables for the MySQL root user password and database name
ENV MYSQL_ROOT_PASSWORD=root
ENV MYSQL_DATABASE=milk_quality_prediction

# Copy the SQL script to the Docker image
COPY milk_quality_prediction.sql /docker-entrypoint-initdb.d/

# Expose the default MySQL port
EXPOSE 3306