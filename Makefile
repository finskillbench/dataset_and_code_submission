.PHONY: verify-submission verify-tables verify-claims manifest clean

# Full verification pipeline
verify-submission: manifest
	python scripts/verify_submission_bundle.py --assert-match

# Individual checks
verify-tables:
	python analysis/reproduce_tables.py --table all

verify-claims:
	python analysis/verify_claims.py --claim all

# Regenerate MANIFEST.sha256
manifest:
	python scripts/build_submission.py

# Clean generated files
clean:
	rm -f MANIFEST.sha256 submission_verification_report.md submission_verification_report.json
