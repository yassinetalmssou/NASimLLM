#!/usr/bin/env bash
################################################################################
# hpc_sync_results.sh — Sync experiment results from VSC_SCRATCH to VSC_DATA
#
# Run this manually or set it up as a cron job on the HPC login node:
#
#   # Open crontab editor:
#   crontab -e
#
#   # Add a line to run every 4 hours (adjust path as needed):
#   0 */4 * * * bash $VSC_DATA/NASimLLM/hpc_sync_results.sh >> $VSC_DATA/sync_results.log 2>&1
#
# Manual usage:
#   bash hpc_sync_results.sh
#
# What it does:
#   Copies all results from $VSC_SCRATCH/runs/ → $VSC_DATA/runs/
#   Uses rsync --update so it only transfers new/changed files.
#   The SCRATCH copy is kept intact (no deletion).
################################################################################

set -e

SCRATCH_RUNS="${VSC_SCRATCH}/runs"
DATA_RUNS="${VSC_DATA}/runs"

echo "========================================================================"
echo "NASimLLM Results Sync  —  $(date)"
echo "========================================================================"
echo "  Source : $SCRATCH_RUNS"
echo "  Target : $DATA_RUNS"
echo ""

if [ ! -d "$SCRATCH_RUNS" ]; then
    echo "Nothing to sync: $SCRATCH_RUNS does not exist (no jobs run yet)."
    exit 0
fi

mkdir -p "$DATA_RUNS"

# --update   : skip files that are newer on the destination (safe for reruns)
# --archive  : preserve permissions, timestamps, symlinks, recursive
# --info=stats2 : print a brief summary of what was transferred
rsync --archive --update --info=stats2 "$SCRATCH_RUNS/" "$DATA_RUNS/"

echo ""
echo "Sync complete."
echo "  SCRATCH : $(du -sh "$SCRATCH_RUNS" 2>/dev/null | cut -f1)"
echo "  DATA    : $(du -sh "$DATA_RUNS"    2>/dev/null | cut -f1)"
echo "========================================================================"
