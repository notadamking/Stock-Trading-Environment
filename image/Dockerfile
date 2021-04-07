# Build from ubuntu base image
FROM stablebaselines/stable-baselines

# Install packages
RUN apt-get -y update \
    && apt-get -y install \
    ffmpeg \
    freeglut3-dev \
    swig \
    vim \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup application in /app
WORKDIR /app

# Make data and environment directories

# Install libraries
RUN pip uninstall -y stable-baselines && \
    pip install stable-baselines[mpi]==2.10.0 && \
    pip install pandas && \
    rm -rf $HOME/.cache/pip

# Move code files into working dir
COPY main.py /app/main.py
COPY env/StockTradingEnv.py /app/env/StockTradingEnv.py
COPY data/AAPL.csv /app/data/AAPL.csv
