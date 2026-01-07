# Optional (placeholder): pin OS + drivers as needed when you start ManiSkill2 install.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY configs/ /app/configs/

RUN pip install --no-cache-dir -U pip setuptools wheel

# Install your real deps once you decide your stack (mani_skill2, torch, gymnasium, sb3, etc.)
# RUN pip install -e ".[dev]"

CMD ["python", "-c", "print('Container scaffold ready. Add dependencies + entrypoints.')"]


