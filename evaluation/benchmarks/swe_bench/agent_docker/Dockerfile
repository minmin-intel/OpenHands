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
COPY ${OPENHANDS_PATH:-OpenHands} /app/OpenHands


# Install OpenHands
WORKDIR /app/OpenHands
# RUN git checkout benchmark-dev

# Install OpenHands in development mode so Python can find it
RUN pip install -e .

RUN make build

# # Install poetry
# RUN pip install poetry

# # Configure poetry to not create a virtual environment
# RUN poetry config virtualenvs.create false

# # Install OpenHands dependencies using poetry
# RUN poetry install --no-dev

# Set the working directory for the server
WORKDIR /app/OpenHands/evaluation/benchmarks/swe_bench/agent_docker

# Install additional dependencies for the API server
COPY OpenHands/evaluation/benchmarks/swe_bench/agent_docker/requirements.txt ./
COPY OpenHands/evaluation/benchmarks/swe_bench/agent_docker/docker_helper.sh ./
RUN pip install -r requirements.txt && \
    chmod +x docker_helper.sh

# Expose the default FastAPI port
EXPOSE 8000

# The Docker group will be created at runtime with the correct GID

# Command to run the FastAPI server with Docker group setup
CMD ["sh", "-c", "if [ ! -z \"$DOCKER_GID\" ]; then groupadd -g $DOCKER_GID docker && usermod -aG docker root; fi && uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]