# Use a base image with better webcam support
FROM jrottenberg/ffmpeg:4.3-ubuntu

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    wget \
    unzip \
    build-essential \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for model cache
RUN mkdir -p /root/.insightface/models/

# Download and extract required InsightFace model
RUN wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    mkdir -p /root/.insightface/models/buffalo_l && \
    unzip -j buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm buffalo_l.zip

# Expose the port Streamlit runs on
EXPOSE 8501

# Configure Streamlit to run on 0.0.0.0
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"] 