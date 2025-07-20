FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.enableCORS=false"]
