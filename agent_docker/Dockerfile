FROM python:3.12-slim

# Define an argument for the OpenHands path
ARG OPENHANDS_PATH

# Install git and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential \
    wget \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge (for mamba)
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3.sh -b -p /opt/conda && \
    rm Miniforge3.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Install Node.js and Poetry using mamba
RUN mamba install -y conda-forge::nodejs
RUN mamba install -y conda-forge::poetry

# Set working directory
WORKDIR /app

# Copy OpenHands from the build context
# Note: This requires setting the build context to the parent directory of OpenHands
COPY ${OPENHANDS_PATH:-OpenHands}/openhands /app/openhands
COPY ${OPENHANDS_PATH:-OpenHands}/agent_docker/server.py /app/server.py
COPY ${OPENHANDS_PATH:-OpenHands}/agent_docker/requirements.txt /app/requirements.txt
COPY ${OPENHANDS_PATH:-OpenHands}/pyproject.toml /app/pyproject.toml
COPY ${OPENHANDS_PATH:-OpenHands}/poetry.lock /app/poetry.lock
RUN touch README.md

RUN export POETRY_CACHE_DIR && poetry install --no-root && rm -rf $POETRY_CACHE_DIR

# # Install Poetry dependencies
# RUN poetry install --with dev,test,runtime

# Install server dependencies
RUN pip install -r requirements.txt
    
# Ensure server.py is executable
RUN chmod +x server.py

# Expose the default FastAPI port
EXPOSE 8000

# The Docker group will be created at runtime with the correct GID

# Command to run the FastAPI server with Docker group setup using Poetry
CMD ["sh", "-c", "if [ ! -z \"$DOCKER_GID\" ]; then groupadd -g $DOCKER_GID docker && usermod -aG docker root; fi && poetry run python server.py --host 0.0.0.0 --port ${PORT:-8000}"]