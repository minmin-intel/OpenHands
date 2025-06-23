FROM python:3.12-slim

# Install git and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone OpenHands repository
RUN git clone https://github.com/All-Hands-AI/OpenHands.git

# Install OpenHands
WORKDIR /app/OpenHands
# Install poetry
RUN pip install poetry

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

# Install OpenHands dependencies using poetry
RUN poetry install --no-dev

# Set the working directory for the server
WORKDIR /app

# Copy requirements and server code
COPY ./requirements.txt /app/
COPY ./server.py /app/

# Install additional dependencies for the API server
RUN pip install -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Command to run the FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]