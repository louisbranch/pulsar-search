.PHONY: test coverage search

PULSAR_WORKERS ?= 5
PULSAR_INSTRUMENT ?= act
PULSAR_TARGET ?= crab


test:
	pytest -q

coverage:
	pytest -q --cov=pulsar --cov-config=.coveragerc --cov-report xml:tests/coverage.xml

search:
	@NUM_PROCESSES=$$(( $(PULSAR_WORKERS) + 1 )); \
    echo "Searching on $(PULSAR_INSTRUMENT) data for $(PULSAR_TARGET) using $$NUM_PROCESSES processes"; \
    mpirun -n $$NUM_PROCESSES -x OMP_NUM_THREADS=1 python3 data/$(PULSAR_INSTRUMENT)/scripts/$(PULSAR_TARGET)_search.py
