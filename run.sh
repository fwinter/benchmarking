#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Wilson DSlash benchmark runner for qdp-jit / Chroma test code
#
# Single GPU:
#   mpirun -np 1 ./t_dslashm -timing-run -geom 1 1 1 1 -lat N
#
# Four GPUs:
#   mpirun -np 4 ./t_dslashm -timing-run -geom 1 1 1 4 -lat N
#
# Output:
#   results_dslash/summary.csv
#   results_dslash/logs/*.log
# ------------------------------------------------------------

EXE="${EXE:-./t_dslashm}"
OUTDIR="${OUTDIR:-results_dslash}"
LOGDIR="${OUTDIR}/logs"
CSV="${OUTDIR}/summary.csv"

# Lattice sizes to test.
SIZES=(8 16 32 64 96)

# Number of repeated program launches per size/config.
# Keep this at 1 initially because each run already reports 4 DSlash timings.
REPEATS="${REPEATS:-1}"

mkdir -p "${LOGDIR}"

echo "config,np,geom,lat,repeat,avg_gflops,min_gflops,max_gflops,num_dslash,logfile" > "${CSV}"

parse_log() {
    local logfile="$1"

    python3 - "$logfile" <<'PY'
import sys
import re
from statistics import mean

logfile = sys.argv[1]

values = []

pattern = re.compile(r"performance\s*=\s*([0-9]+(?:\.[0-9]*)?)\s*GFlops")

with open(logfile, "r", errors="replace") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            values.append(float(m.group(1)))

if not values:
    print("nan,nan,nan,0")
else:
    print(f"{mean(values):.6f},{min(values):.6f},{max(values):.6f},{len(values)}")
PY
}

run_case() {
    local config="$1"
    local np="$2"
    local geom="$3"
    local lat="$4"
    local repeat="$5"

    local safe_geom
    safe_geom="$(echo "${geom}" | tr ' ' 'x')"

    local logfile="${LOGDIR}/dslash_${config}_np${np}_geom${safe_geom}_lat${lat}_rep${repeat}.log"

    echo "Running: config=${config}, np=${np}, geom=${geom}, lat=${lat}, repeat=${repeat}"
    echo "Log: ${logfile}"

    set +e
    mpirun -np "${np}" "${EXE}" -timing-run -geom ${geom} -lat "${lat}" > "${logfile}" 2>&1
    local status=$?
    set -e

    if [[ "${status}" -ne 0 ]]; then
        echo "ERROR: run failed with exit code ${status}. See ${logfile}" >&2
        echo "${config},${np},\"${geom}\",${lat},${repeat},nan,nan,nan,0,${logfile}" >> "${CSV}"
        return
    fi

    local parsed
    parsed="$(parse_log "${logfile}")"

    local avg min max count
    IFS=',' read -r avg min max count <<< "${parsed}"

    if [[ "${count}" -ne 4 ]]; then
        echo "WARNING: expected 4 DSlash performance lines, found ${count} in ${logfile}" >&2
    fi

    echo "${config},${np},\"${geom}\",${lat},${repeat},${avg},${min},${max},${count},${logfile}" >> "${CSV}"
}

for lat in "${SIZES[@]}"; do
    for rep in $(seq 1 "${REPEATS}"); do
        run_case "single_gpu" 1 "1 1 1 1" "${lat}" "${rep}"
        run_case "four_gpu"   4 "1 1 1 4" "${lat}" "${rep}"
    done
done

echo
echo "Done."
echo "Summary CSV: ${CSV}"
echo "Logs:        ${LOGDIR}"

