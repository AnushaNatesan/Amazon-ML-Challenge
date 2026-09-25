#!/usr/bin/env python3
"""
Packaging and Validation Utility for Official Challenge Submission
Team: GenX H4CK3RS!

Builds GenX_H4CK3RS!_submission.zip matching the required structure:
GenX_H4CK3RS!_submission.zip/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

TEAM_NAME = "GenX_H4CK3RS!"
ZIP_NAME = f"{TEAM_NAME}_submission.zip"


def create_submission_zip(workspace_dir: Path) -> Path:
    """Creates the submission zip archive containing output/, code/, and Documentation_template.md."""
    zip_path = workspace_dir / ZIP_NAME
    output_dir = workspace_dir / "output"
    code_dir = workspace_dir / "code" / "business_entity_resolution"
    doc_file = workspace_dir / "Documentation_template.md"

    print(f"[*] Packaging submission for team '{TEAM_NAME}'...")
    print(f"[*] Destination: {zip_path}")

    # Ensure output files exist
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    if not matching_file.exists():
        print(f"[!] Warning: {matching_file} does not exist yet. Please run inference to generate final predictions.")
    if not candidate_file.exists():
        print(f"[!] Warning: {candidate_file} does not exist yet. Please run candidate blocking to generate candidates.")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add output files if present
        if output_dir.exists():
            for f in output_dir.glob("*.tsv"):
                arcname = f"output/{f.name}"
                zipf.write(f, arcname=arcname)
                print(f"  + Added: {arcname}")

        # Add Documentation_template.md
        if doc_file.exists():
            zipf.write(doc_file, arcname="Documentation_template.md")
            print("  + Added: Documentation_template.md")

        # Add code/business_entity_resolution directory
        for root, dirs, files in os.walk(code_dir):
            # Ignore cache and tmp directories
            dirs[:] = [d for d in dirs if d not in {"__pycache__", ".pytest_cache", "tmp_test_data"}]
            for file in files:
                if file.endswith((".pyc", ".pyo")):
                    continue
                file_path = Path(root) / file
                rel_path = file_path.relative_to(code_dir)
                arcname = f"code/business_entity_resolution/{rel_path.as_posix()}"
                zipf.write(file_path, arcname=arcname)
                print(f"  + Added: {arcname}")

    print(f"[+] Successfully built: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    return zip_path


def validate_against_official_tool(workspace_dir: Path):
    """Executes official validator if output files are populated."""
    validator = workspace_dir / "Dataset" / "student_resource" / "utils" / "validate_submission.py"
    test_dir = workspace_dir / "Dataset" / "student_resource" / "dataset" / "test"
    matching_file = workspace_dir / "output" / "matching_results.tsv"
    candidate_file = workspace_dir / "output" / "candidate_pairs.tsv"

    if not validator.exists():
        print("[!] Official validator script not found.")
        return

    if not matching_file.exists() or matching_file.stat().st_size == 0:
        print("[*] Output matching_results.tsv is empty or not yet generated. Validation check skipped for now.")
        return

    cmd = [
        sys.executable,
        str(validator),
        "--matching", str(matching_file),
        "--candidate", str(candidate_file),
        "--test-dir", str(test_dir),
    ]

    print("\n[*] Running official competition validator...")
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("[+] Submission files PASSED official competition validation!")
    else:
        print(f"[!] Official validator returned exit code {res.returncode}.")


if __name__ == "__main__":
    workspace = Path(__file__).resolve().parent.parent.parent
    create_submission_zip(workspace)
    validate_against_official_tool(workspace)
