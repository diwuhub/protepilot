# PharmaDev AI — Makefile
# ========================
# Usage:
#   make check          # Full quality gate (contracts + benchmarks + audit)
#   make check-fast     # Contracts + audit only (no benchmarks)
#   make contracts      # All contract suites
#   make benchmarks     # All benchmark suites
#   make audit          # Dependency audit only
#   make versions       # Show all package versions

PYTHON ?= python

.PHONY: check check-fast contracts benchmarks audit versions help \
       contract-twin contract-report-bulk contract-auxiliary contract-infra contract-medium \
       benchmark-twin benchmark-report-bulk benchmark-auxiliary benchmark-infra benchmark-medium

help:
	@echo "PharmaDev AI — Available targets:"
	@echo "  make check        Full quality gate (all suites)"
	@echo "  make check-fast   Contracts + audit (skip benchmarks)"
	@echo "  make contracts    All contract suites"
	@echo "  make benchmarks   All benchmark suites"
	@echo "  make audit        Dependency audit"
	@echo "  make versions     Show package versions"

check:
	$(PYTHON) run_all_checks.py

check-fast:
	$(PYTHON) run_all_checks.py --fast

contracts: contract-twin contract-report-bulk contract-auxiliary contract-infra contract-medium

contract-twin:
	$(PYTHON) -m src.twin_contracts

contract-report-bulk:
	$(PYTHON) -m src.report_bulk_contracts

contract-auxiliary:
	$(PYTHON) -m src.auxiliary_contracts

contract-infra:
	$(PYTHON) -m src.infra_contracts

contract-medium:
	$(PYTHON) -m src.medium_contracts

benchmarks: benchmark-twin benchmark-report-bulk benchmark-auxiliary benchmark-infra benchmark-medium

benchmark-twin:
	$(PYTHON) -m src.twin_benchmark

benchmark-report-bulk:
	$(PYTHON) -m src.report_bulk_benchmark

benchmark-auxiliary:
	$(PYTHON) -m src.auxiliary_benchmark

benchmark-infra:
	$(PYTHON) -m src.infra_benchmark

benchmark-medium:
	$(PYTHON) -m src.medium_benchmark

audit:
	$(PYTHON) -m src.dependency_audit

versions:
	$(PYTHON) packages/bump_version.py --show
