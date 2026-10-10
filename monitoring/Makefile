# Monitoring component convenience targets.
# Run from the repository root:  make -f monitoring/Makefile test
VENV ?= .venv
PY ?= $(VENV)/bin/python

.PHONY: test demo lint clean-demo

test:
	$(PY) -m pytest monitoring/tests -q

demo:
	$(PY) -m monitoring --out-dir out/monitoring --months 6

lint:
	$(PY) -m compileall -q monitoring

clean-demo:
	rm -rf out/monitoring
