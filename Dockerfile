FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    XDG_CACHE_HOME=/tmp/genesis-lab-cache \
    NUMBA_CACHE_DIR=/tmp/genesis-lab-cache/numba \
    GS_CACHE_FILE_PATH=/tmp/genesis-lab-cache/genesis \
    QD_OFFLINE_CACHE_FILE_PATH=/tmp/genesis-lab-cache/quadrants \
    PYOPENGL_PLATFORM=egl \
    EGL_PLATFORM=surfaceless \
    LIBGL_ALWAYS_SOFTWARE=1 \
    OMP_NUM_THREADS=4 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1 libgl1-mesa-dri libosmesa6 libglib2.0-0 \
    libgomp1 libxrender1 libxext6 ffmpeg ca-certificates fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip check
RUN useradd --create-home --uid 1000 learner \
    && mkdir -p /tmp/genesis-lab-cache/numba /home/learner/.cache \
    && chown -R learner:learner /workspace /tmp/genesis-lab-cache /home/learner/.cache
COPY --chown=learner:learner . /workspace
USER learner
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.root_dir=/workspace"]
