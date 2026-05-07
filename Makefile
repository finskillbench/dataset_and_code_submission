PYTHON ?= python3.12

.PHONY: verify-submission verify-tables verify-claims verify-expected verify-tests manifest clean

# Full verification pipeline
verify-submission: manifest
	$(PYTHON) scripts/verify_submission_bundle.py --assert-match

# Individual checks
verify-tables:
	$(PYTHON) analysis/reproduce_tables.py --table all

verify-claims:
	$(PYTHON) analysis/verify_claims.py --claim all

verify-expected:
	$(PYTHON) analysis/verify_expected_results.py

verify-tests:
	$(PYTHON) -m pytest evaluation/finskillbench_agent/tests evaluation/hermes_results/scoring/tests

# Regenerate MANIFEST.sha256
manifest:
	$(PYTHON) scripts/build_submission.py

# Clean generated files
clean:
	rm -f MANIFEST.sha256 submission_verification_report.md submission_verification_report.json
