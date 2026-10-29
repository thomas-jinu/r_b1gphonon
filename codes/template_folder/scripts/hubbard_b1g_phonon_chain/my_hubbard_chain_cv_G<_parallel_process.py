"""Collect frequency-resolved operator output from a stepped DMRG++ run.

Expected layout:
    RUN_FOLDER/
      omega_axis_*.txt
      step_0000/runForinput_*_apply.cout
      step_0001/runForinput_*_apply.cout
      ...

Output records have the form:
    site real_part imaginary_part <gs|c'|P2> time_or_other_trailing_value

The frequency is taken from omega_axis[step_number], not from the final
number printed on the operator record.
"""

import argparse
import csv
import re
from pathlib import Path

TWOTIME_OPERATOR_LABEL = "<gs|c'|P2>"
TWOTIME_COLLECTION_MARKER = "FiniteLoops printing ends"
NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
OPERATOR_PATTERN = re.compile(
    rf"^\s*(\d+)\s+({NUMBER_PATTERN})\s+({NUMBER_PATTERN})\s+"
    rf"{re.escape(TWOTIME_OPERATOR_LABEL)}\s+({NUMBER_PATTERN})\s*$"
)


def read_operator_output(output_path: Path) -> dict[int, complex]:
    """Read site -> complex value after the collection marker; last wins."""
    values: dict[int, complex] = {}
    collecting = False
    with output_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if TWOTIME_COLLECTION_MARKER in line:
                collecting = True
                continue
            if not collecting or TWOTIME_OPERATOR_LABEL not in line:
                continue
            match = OPERATOR_PATTERN.match(line)
            if match:
                site = int(match.group(1))
                values[site] = complex(float(match.group(2)), float(match.group(3)))
    return values


def find_omega_file(run_folder: Path) -> Path:
    candidates = sorted(run_folder.glob("omega_axis_*.txt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one omega_axis_*.txt in {run_folder}; found {len(candidates)}"
        )
    return candidates[0]


def collect_run(run_folder: Path, center_site: int, output_path: Path) -> int:
    omega_file = find_omega_file(run_folder)
    omega_axis = [float(x) for x in omega_file.read_text().split()]
    step_dirs = sorted(
        (p for p in run_folder.glob("step_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not step_dirs:
        raise FileNotFoundError(f"No step_* directories in {run_folder}")

    rows = []
    for step_dir in step_dirs:
        try:
            step_index = int(step_dir.name.split("_")[-1])
        except ValueError as error:
            raise ValueError(f"Invalid step directory name: {step_dir.name}") from error
        if step_index >= len(omega_axis):
            raise ValueError(
                f"{step_dir.name} maps to omega index {step_index}, but axis has "
                f"only {len(omega_axis)} values"
            )

        outputs = sorted(step_dir.glob("runForinput_*_apply.cout"))
        if len(outputs) != 1:
            raise FileNotFoundError(
                f"Expected one runForinput_*_apply.cout in {step_dir}; found {len(outputs)}"
            )
        omega = omega_axis[step_index]
        for site, value in read_operator_output(outputs[0]).items():
            rows.append((omega, center_site, site, value.real, value.imag))

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["omega", "center_site", "site_index", "real_part", "imaginary_part"]
        )
        for omega, center, site, real, imag in sorted(rows):
            writer.writerow(
                [f"{omega:.15g}", center, site, f"{real:.15g}", f"{imag:.15g}"]
            )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect frequency-resolved DMRG++ operator results from step folders."
    )
    parser.add_argument(
        "run_folder", type=Path, help="Folder containing omega_axis and step_####"
    )
    parser.add_argument("center_site", type=int, help="Center site index")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV (default: RUN_FOLDER/collected_operator_data.csv)",
    )
    args = parser.parse_args()

    run_folder = args.run_folder.expanduser().resolve()
    if not run_folder.is_dir():
        raise FileNotFoundError(f"Run folder does not exist: {run_folder}")
    output_path = args.output or run_folder / "collected_operator_data.csv"
    count = collect_run(run_folder, args.center_site, output_path)
    print(f"Wrote {count} records to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
