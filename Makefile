.PHONY: test smoke train

test:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q

smoke:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 scripts/make_synthetic_data.py --output data --dims 16 16 16 --count 24
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m torr.train --config configs/smoke.yaml
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m torr.evaluate --checkpoint checkpoints/smoke.pt --data data/test.pt

train:
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m torr.train --config configs/default.yaml

