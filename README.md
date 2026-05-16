# Multi-methods Colorization — CV 2025.2

Ba phương pháp tô màu ảnh: **Scribble-based** (Levin 2004) · **Example-based** (Welsh 2002) · **Deep Learning** (Zhang 2016).

---

## Setup

Yêu cầu: [Docker Desktop] đang chạy.

```bash
# Build image (lần đầu hoặc khi requirements.txt thay đổi)
docker compose -f docker/docker-compose.yml build

# Shell dev
docker compose -f docker/docker-compose.yml run --rm dev

# Gradio demo  →  http://localhost:7860
docker compose -f docker/docker-compose.yml up demo

# Tests
docker compose -f docker/docker-compose.yml run --rm dev pytest tests/ -v

# Jupyter  →  http://localhost:8888
docker compose -f docker/docker-compose.yml run --rm dev jupyter lab --ip 0.0.0.0 --no-browser
```

> GPU training chạy local (không qua Docker). Xem `MASTER_PLAN.md` để biết chi tiết.
