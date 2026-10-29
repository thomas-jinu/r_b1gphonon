"""
Generate and optionally run a two-time Hubbard-chain calculation.
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Location of the current DMRG++ executables for different clusters. Adjust these paths as needed. The location is printed when the script is completed.
DMRG_EXECUTABLES = {
    "local": Path(
        "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
    ),
    "nersc": Path("/global/common/software/m5228/dmrgpp/installdir/bin/dmrg"),
    # "nersc": Path(
    #     "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
    # ),
}


# DMRG Settings
DMRG_PRECISION = 12

singletime_cv_OPERATOR_LABEL = "<gs|c'|P2>"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a DMRG++ Hubbard-chain calculation.",
    )
    parser.add_argument("sites", type=int, help="Chain length N")
    parser.add_argument("up", type=int, help="Number of spin-up electrons")
    parser.add_argument("down", type=int, help="Number of spin-down electrons")
    parser.add_argument("t", type=float, help="Hopping parameter t")
    parser.add_argument("U", type=float, help="Hubbard interaction U")
    parser.add_argument("potentialV", type=float, help="Potential V")
    parser.add_argument("gs_filename", type=Path, help="Ground-state file location")
    parser.add_argument(
        "finite_kept", type=int, help="Number of states in finite loops"
    )
    parser.add_argument(
        "TSPAdvanceEach",
        type=int,
        help="Multiples of N-2",
    )
    parser.add_argument("eta_broadening", type=float)
    parser.add_argument("omega_in", type=float)
    parser.add_argument("omega_step_size", type=float)
    parser.add_argument("omega_steps", type=int)
    parser.add_argument("cluster", choices=DMRG_EXECUTABLES)
    parser.add_argument("center_site", type=int, help="Index of the center site")
    parser.add_argument(
        "finite_loops_cv",
        type=int,
        help="Number of finite loops for applying the operator step",
    )
    parser.add_argument(
        "restart_filename",
        type=Path,
        help="Either gs or some timedependent restart file",
    )
    parser.add_argument(
        "time_step_start_index", type=int, help="Starting time step index for the run"
    )
    parser.add_argument(
        "--run", action="store_true", help="Run DMRG++ after generating inputs"
    )
    parser.add_argument(
        "--submit", action="store_true", help="Submit DMRG++ job after generating input"
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="CPU cores assigned to each DMRG++ process.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=2,
        help=(
            "Number of independent time steps to run simultaneously. "
            "Must be at least 2 so the first and last steps can run together."
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate constraints required by the finite-loop construction."""
    if args.sites <= 2:
        raise ValueError("sites must be greater than 2")
    if args.TSPAdvanceEach <= 0:
        raise ValueError("TSPAdvanceEach must be positive")
    if args.TSPAdvanceEach % (args.sites - 2) != 0:
        raise ValueError("TSPAdvanceEach must be a multiple of sites - 2")
    if args.eta_broadening <= 0:
        raise ValueError("eta_broadening must be positive")
    if args.n_tasks < 2:
        raise ValueError("n_tasks must be at least 2")


def save_arguments(args: argparse.Namespace, output_path: Path) -> None:
    """Save command-line arguments as JSON."""
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(values, file, indent=2)


def build_input_apply(
    args: argparse.Namespace,
    run_name: str,
    omega_axis: list[float],
    step: int,
    restart_filename_apply: str,
) -> str:
    """Build the input that applies the operator at one time step."""

    # Finite rows for the apply step.
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 2]" for _ in range(args.finite_loops_cv)
    )

    return (
        "\n\n".join(
            [
                "##Ainur1.0",
                "\n".join(
                    [
                        "# --- Model parameters ---",
                        f"TotalNumberOfSites = {args.sites};",
                        "NumberOfTerms = 1;",
                        "DegreesOfFreedom = 1;",
                        'GeometryKind = "chain";',
                        'GeometryOptions = "ConstantValues";',
                        f"dir0:Connectors = [{args.t}];",
                        f"hubbardU = [{args.U}, ...];",
                        f"potentialV = [{args.potentialV}, ...];",
                        'Model = "HubbardOneBand";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Fock Space parameters ---",
                        f"TargetElectronsUp = {args.up};",
                        f"TargetElectronsDown = {args.down};",
                    ]
                ),
                "\n".join(  # noqa: FLY002
                    [
                        "# --- DMRG++ control parameters ---",
                        "TruncationTolerance = 1e-12;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "CorrectionVectorTargeting,restart,minimizeDisk,restart,twositedmrg";',
                        'Version = "stc_vs_ttc_apply";',
                        f"CorrectionVectorOmega = {omega_axis[step]};",
                        "DynamicDmrgType = 1;",
                        'TSPProductOrSum = "sum";',
                        f"TSPSites = [{args.center_site}];",
                        "TSPLoops = [1];",
                        'string TSPOp0:TSPOperator = "expression";',
                        'string TSPOp0:OperatorExpression = "c";',
                        "CorrectionA = 0;",
                        'CorrectionVectorFreqType = "Real";',
                        f"CorrectionVectorEta = {args.eta_broadening};",
                        'CorrectionVectorAlgorithm = "Krylov";',
                        f'OutputFile = "{run_name}_apply";',
                        f'RestartFilename = "{restart_filename_apply}";',
                        "GsWeight = 0.1;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def run_dmrg(
    working_directory: Path,
    input_path: Path,
    operator: str | None = None,
    cpus_per_task: int = 1,
) -> None:
    dmrg_command = [
        "./dmrg",
        "-f",
        input_path.name,
        "-p",
        str(DMRG_PRECISION),
    ]

    if operator is not None:
        dmrg_command.append(operator)

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(cpus_per_task)

    subprocess.run(
        dmrg_command,
        cwd=working_directory,
        check=True,
        env=environment,
    )


def run_one_step(
    step_folder: Path,
    input_path_apply: Path,
    cpus_per_task: int,
) -> None:
    """Run the apply and evolve calculations for one time step."""
    run_dmrg(
        step_folder,
        input_path_apply,
        operator=singletime_cv_OPERATOR_LABEL,
        cpus_per_task=cpus_per_task,
    )


def main() -> int:
    """Generate inputs, optionally run DMRG++, and collect the results."""
    args = parse_args()
    restart_path = args.gs_filename.resolve()

    omega_axis = args.omega_in + args.omega_step_size * np.arange(
        args.omega_steps, dtype=float
    )

    omega_axis = np.round(omega_axis, decimals=5)
    omega_axis = omega_axis.tolist()

    run_name = args.restart_filename.stem
    print(f"Run name: {run_name}")
    executable_path = DMRG_EXECUTABLES[args.cluster]

    if not restart_path.is_file():
        raise FileNotFoundError(f"Ground-state file not found: {restart_path}")
    if not executable_path.is_file():
        raise FileNotFoundError(f"DMRG++ executable not found: {executable_path}")

    # Name file with pump frequency for clarity
    run_folder_singletime_cv = (
        restart_path.parent
        / f"G<_singletime_cv_{run_name}_steps={args.time_step_start_index}"
    )

    run_folder_singletime_cv.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder_singletime_cv}")

    np.savetxt(
        run_folder_singletime_cv / f"omega_axis_{args.restart_filename.stem}.txt",
        omega_axis,
        delimiter=",",
        fmt="%.15g",
    )

    # Save the command-line arguments to a JSON file.
    args_path = run_folder_singletime_cv / f"input_args_singletime_cv_{run_name}.json"
    save_arguments(args, args_path)
    print(f"Wrote arguments: {args_path}")

    step_jobs = []

    for step in range(args.omega_steps):
        step_folder = run_folder_singletime_cv / f"step_{step:04d}"
        step_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created step folder: {step_folder}")
        shutil.copy2(executable_path, step_folder / "dmrg")

        input_path_apply = step_folder / f"input_singletime_cv_{run_name}_apply.ain"
        input_path_apply.write_text(
            build_input_apply(
                args,
                f"singletime_cv_{run_name}",
                step=step,
                omega_axis=omega_axis,
                restart_filename_apply=f"../../{run_name}",
            ),
            encoding="utf-8",
        )

        print(f"Wrote Ainur input: {input_path_apply}")

        step_jobs.append((step_folder, input_path_apply))

    # step_jobs remains in chronological order: 0, 1, 2, ..., final step.
    if args.cluster == "nersc":
        slurm_path = run_folder_singletime_cv / f"batch_{run_name}.slurm"

        # Assuming two hardware threads per physical core; never request zero
        # OpenMP threads when cpus_per_task is 1.
        physical_cores_per_task = max(1, args.cpus_per_task // 2)

        body = f"""#!/bin/bash
#SBATCH --account=m5228
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks={args.n_tasks}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --time=48:00:00
#SBATCH --job-name=dmrg_singletime_cv_{run_name}
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load PrgEnv-gnu/8.7.0
module load cray-mpich/9.1.0
module load cray-libsci/26.03.0
module load cray-hdf5/1.14.3.7

export CC=cc
export CXX=CC

export SLURM_CPU_BIND="cores"

export OMP_NUM_THREADS={physical_cores_per_task}
export OMP_PLACES=cores   
export OMP_PROC_BIND=spread

export BASE=/global/common/software/m5228
export LOCAL="$BASE/local"


date

# Start time steps from beginning to end. Refill each slot as soon as any step
# finishes; do not wait for all running steps, which would leave CPUs idle.
step_folders=(
{chr(10).join(f'    "{job[0]}"' for job in step_jobs)}
)
active=0
failed=0

for step_folder in "${{step_folders[@]}}"; do
    (
        cd "$step_folder"

        srun \\
            --exclusive \\
            --ntasks=1 \\
            --cpus-per-task={args.cpus_per_task} \\
            ./dmrg \\
            -f "input_singletime_cv_{run_name}_apply.ain" \\
            -p "{DMRG_PRECISION}"

    ) &

    active=$((active + 1))

    # Once all slots are occupied, wait only for the next job to finish and
    # immediately use its freed slot for the next chronological step.
    if (( active >= {args.n_tasks} )); then
        if ! wait -n; then
            failed=1
        fi
        active=$((active - 1))
    fi
done

# Reap jobs still running after all time steps have been launched.
while (( active > 0 )); do
    if ! wait -n; then
        failed=1
    fi
    active=$((active - 1))
done

if (( failed != 0 )); then
    echo "One or more two-time steps failed." >&2
    exit 1
fi

date
"""

        slurm_path.write_text(body, encoding="utf-8")
        print(f"Wrote batch script: {slurm_path}")

    if args.run:
        with ThreadPoolExecutor(max_workers=args.n_tasks) as executor:
            futures = [
                executor.submit(
                    run_one_step,
                    step_folder,
                    input_path_apply,
                    args.cpus_per_task,
                )
                for step_folder, input_path_apply in step_jobs
            ]

            for future in as_completed(futures):
                future.result()

    elif args.submit:
        subprocess.run(
            ["sbatch", str(slurm_path)],
            cwd=run_folder_singletime_cv,
            check=True,
        )
    else:
        print("Inputs generated. Use --run to execute DMRG++.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
