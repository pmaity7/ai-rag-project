# Dockerfile

# Step 1 — Start from official Python 3.11 slim image
FROM python:3.11-slim

# Step 2 — Set working directory inside the container
WORKDIR /app

# Step 3 — Copy requirements and install dependencies
# Doing this before copying code takes advantage of Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4 — Copy the rest of the project files
COPY . .

# Step 5 — Expose Streamlit's default port
EXPOSE 8501

# Step 6 — Command to run when the container starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]