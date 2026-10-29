"""Generate and optionally run a DMRG++ Hubbard-chain ground state.

Run this script from above the directory containing the ``dmrgpp`` scripts,
or adjust the project-root discovery below if your package layout differs.
"""

import argparse
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Resolve paths so that scripts live inside scripts/hubbard_chain, and the project root is two levels up.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # two level up to the project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


# Define all the input arguments for the script, including the cluster choice and the --run flag.
def parse_args_input() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a DMRG++ Hubbard-chain calculation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Order of positional arguments: sites up down t U potentialV infinite_kept finite_loops finite_kept cluster

        Example:
        python make_hubbard.py 32 16 16 -1 8 0 128 5 1000 local --run
        """,
    )
    parser.add_argument("sites", type=int, help="Chain length N")
    parser.add_argument("up", type=int)
    parser.add_argument("down", type=int)
    parser.add_argument("t", type=float)
    parser.add_argument("U", type=float)
    parser.add_argument("potentialV", type=float)
    parser.add_argument("infinite_kept", type=int)
    parser.add_argument("finite_loops", type=int)
    parser.add_argument("finite_kept", type=int)
    parser.add_argument("cluster", choices=["local", "isaac", "nersc"])
    parser.add_argument(
        "--run", action="store_true", help="Run DMRG++ after generating input"
    )
    parser.add_argument(
        "--submit", action="store_true", help="Submit DMRG++ job after generating input"
    )
    return parser.parse_args()


def build_input(args: argparse.Namespace, up: int, down: int, run_name: str) -> str:
    """Build the Ainur input file content based on the provided arguments."""
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, @save]" for _ in range(args.finite_loops)
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
                        f"TargetElectronsUp = {up};",
                        f"TargetElectronsDown = {down};",
                        'DefineOperators="double:nup*ndown,hole:identity+(-1.0)*nup+(-1.0)*ndown+nup*ndown,parity:identity+(-2.0)*n+4.0*nup*ndown,local_moment:0.75*n+(-1.5)*nup*ndown";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- DMRG++ control parameters ---",
                        f"InfiniteLoopKeptStates = {args.infinite_kept};",
                        "TruncationTolerance = 1e-12;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg";',
                        'Version = "stc_vs_ttc";',
                        f'OutputFile = "{run_name}";',
                    ]
                ),
            ]
        )
        + "\n"
    )


def main() -> int:
    args = parse_args_input()

    # Create the run name based on the input parameters
    # Prepare the run folder and executable path.

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"N={args.sites}_Nup={args.up}_Ndown={args.down}_{timestamp}"
    run_folder = PROJECT_ROOT / run_name
    run_folder.mkdir(parents=True, exist_ok=False)
    print(f"Created run folder: {run_folder}")

    executable_path = DMRG_EXECUTABLES[args.cluster]
    print(f"Using DMRG++ executable for cluster '{args.cluster}': {executable_path}")
    shutil.copy2(executable_path, run_folder / "dmrg")

    # Create input file for Ainur and write it to the run folder.
    input_path = run_folder / f"input_{run_name}.ain"
    input_path.write_text(
        build_input(args, args.up, args.down, run_name), encoding="utf-8"
    )
    print(f"Wrote Ainur input: {input_path}")

    # Generate batch script for NERSC if the cluster is set to 'nersc'.
    if args.cluster == "nersc":
        slurm_path = run_folder / f"batch_{run_name}.slurm"
        body = f"""#!/bin/bash
#SBATCH --account=m5228
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --job-name=dmrg_job
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load PrgEnv-gnu/8.7.0
module load cray-mpich/9.1.0
module load cray-libsci/26.03.0
module load cray-hdf5/1.14.3.7

export CC=cc
export CXX=CC
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export BASE=/global/common/software/m5228
export LOCAL="$BASE/local"

cd "{run_folder}"

date
srun ./dmrg -f "{input_path.name}" -p "{DMRG_PRECISION}"
date
"""
        slurm_path.write_text(body, encoding="utf-8")
        print(f"Wrote batch script: {slurm_path}")

    # Run DMRG++ if the --run flag is provided, otherwise just generate the input files.
    if args.run:
        subprocess.run(
            ["./dmrg", "-f", str(input_path), "-p", f"{DMRG_PRECISION}"],
            cwd=run_folder,
            check=True,
        )
    elif args.submit:
        subprocess.run(
            ["sbatch", str(slurm_path)],
            cwd=run_folder,
            check=True,
        )
    else:
        print("Input generated. Use --run to start DMRG++.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
