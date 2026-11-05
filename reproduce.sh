#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  export PATH="$ROOT/.venv/bin:$PATH"
fi
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache"

QUICK=0
case "${1:-}" in
  --quick) QUICK=1 ;;
  "") ;;
  *) echo "usage: $0 [--quick]" >&2; exit 2 ;;
esac

run_step() {
  local name="$1"
  shift
  echo
  echo "=== ${name} ==="
  local start
  start="$(date +%s)"
  if "$@"; then
    local elapsed=$(( $(date +%s) - start ))
    echo "=== ${name}: ${elapsed}s ==="
  else
    local status=$?
    local elapsed=$(( $(date +%s) - start ))
    echo "=== ${name}: FAILED after ${elapsed}s (exit ${status}) ===" >&2
    exit "$status"
  fi
}

if [[ "$QUICK" -eq 0 ]]; then
  run_step "data fetch offline" python data/fetch.py --offline
fi

run_step "tests" python -m unittest discover -s tests -v

if [[ "$QUICK" -eq 0 ]]; then
  run_step "approach1 Q1A settle" python code/approach1_q1a_svi.py --price-col settle --out-dir results/approach1/io
  run_step "approach1 Q1A close" python code/approach1_q1a_svi.py --price-col close --out-dir results/approach1/io_close
  # results/approach2/io is a historical run from commit 664a716; only io_scaled is reproduced.
  run_step "approach2 Q1B neural" python code/approach2_q1b_neural.py
  run_step "approach3 Q2A Monte Carlo" python code/approach3_q2a_mc.py
  run_step "approach4 Q2B deep hedge" python code/approach4_q2b_deephedge.py
  run_step "approach5 Q3A classical" python code/approach5_q3a_classical.py
  run_step "approach5 Q3A post-hoc" python code/approach5_q3a_posthoc.py
  run_step "approach6 Q3B sequence" python code/approach6_q3b_sequence.py
  run_step "approach6 Q3B post-hoc" python code/approach6_q3b_posthoc.py --mode all
fi

run_step "compare" python code/compare.py
