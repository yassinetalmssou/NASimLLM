#!/usr/bin/env bash
# Sync experiment results from $VSC_SCRATCH/runs to $VSC_DATA/runs (rsync --update; SCRATCH kept).
# Run manually (bash hpc/sync_results.sh) or from cron, e.g.:
#   0 */4 * * * bash $VSC_DATA/NASimLLM/hpc/sync_results.sh >> $VSC_DATA/sync_results.log 2>&1

set -e

SCRATCH_RUNS="${VSC_SCRATCH}/runs"
DATA_RUNS="${VSC_DATA}/runs"

echo "NASimLLM results sync - $(date)"
echo "  Source : $SCRATCH_RUNS"
echo "  Target : $DATA_RUNS"
echo ""

if [ ! -d "$SCRATCH_RUNS" ]; then
    echo "Nothing to sync: $SCRATCH_RUNS does not exist (no jobs run yet)."
    exit 0
fi

mkdir -p "$DATA_RUNS"

# Sync new/changed files only; keep the SCRATCH copy intact.
rsync --archive --update --info=stats2 "$SCRATCH_RUNS/" "$DATA_RUNS/"

echo ""
echo "Sync complete."
echo "  SCRATCH : $(du -sh "$SCRATCH_RUNS" 2>/dev/null | cut -f1)"
echo "  DATA    : $(du -sh "$DATA_RUNS"    2>/dev/null | cut -f1)"
