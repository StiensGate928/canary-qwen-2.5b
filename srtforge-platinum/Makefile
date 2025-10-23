.PHONY: setup lint format test verify docker-gpu

setup:
bash scripts/setup_env.sh

lint:
black --check src
flake8 src tests
mypy src

format:
black src tests

test:
pytest

verify:
bash scripts/verify_binaries.sh

docker-gpu:
docker build -f docker/Dockerfile.cuda -t srtforge-platinum:cuda .
