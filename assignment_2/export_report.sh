#!/usr/bin/env bash
# Usage: bash assignment_2/export_report.sh [absolute-path-to-pandoc]
set -euo pipefail

pandoc_bin="${1:-pandoc}"
report_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_directory="$report_directory/../output/assignment_2"
mkdir -p "$output_directory"
cd "$report_directory"

"$pandoc_bin" assignment_2.md \
  --standalone --embed-resources --math-method=mathml \
  --css=report.css --metadata=pagetitle:"Assignment 2" \
  --output="$output_directory/assignment_2.html"

printf 'Created %s\n' "$output_directory/assignment_2.html"
printf 'Open it in your browser, then choose Print > Save as PDF.\n'
