
# Use an official Python runtime as a parent image
FROM python:3.8.18

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install PyMySQL[rsa]
RUN python3 -m pip install cryptography

# Copy the rest of the working directory contents into the container
COPY . /app

# Expose port 8501 for the Streamlit app
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Command to run the Streamlit app