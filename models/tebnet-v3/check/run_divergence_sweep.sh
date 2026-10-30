#!/bin/bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
WORK="${WORK:-$HOME/teb_work}"
LOGS="$WORK/logs"
STATUS="$WORK/sweep_status.tsv"
CSV="${CSV:-/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/deeptherm_data.csv}"
TAG="thermo_physics_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h_sp-v2_exact_splitH"
FREE_MB="${FREE_MB:-12000}"
POLL="${POLL:-120}"
GPUS="${GPUS:-0 1}"
MAX_WAIT_H="${MAX_WAIT_H:-24}"
NICE="${NICE:-10}"
mkdir -p "$LOGS"
cd "$PROJ" || exit 1
[ -f "$STATUS" ] || printf 'split\tmember\tseed\tgpu\tstate\tstart\tend\tseconds\n' > "$STATUS"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOGS/sweep_driver.log"; }
free_mb() {
    nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
}
plan=()
for s in 1 2 3 4; do
    for m in $(seq 1 9); do
        d="benchmark_results_deeptherm/$TAG/random$s/member${m}_seed${m}"
        if [ -f "$d/results.json" ]; then
            log "skip random$s member$m (already done)"
            continue
        fi
        plan+=("$s:$m")
    done
done
log "plan: ${#plan[@]} run(s) -> ${plan[*]:-none}"
[ "${#plan[@]}" -eq 0 ] && { log "nothing to do"; exit 0; }
if [ "${DRYRUN:-0}" = "1" ]; then
    for item in "${plan[@]}"; do
        s="${item%%:*}"; m="${item##*:}"
        echo "CUDA_VISIBLE_DEVICES=<gpu> python scripts/train.py --seed $m --member $m --split_seed $s" \
             "--csv $CSV --s_prior v2_exact --loss_weights 1 0 0 --monitor h --tag_suffix _splitH --overwrite"
    done
    exit 0
fi
start_all=$(date +%s)
limit=$(( MAX_WAIT_H * 3600 ))
declare -A pid_of
wait_for_slot() {
    local deadline=$(( $(date +%s) + limit ))
    while :; do
        for g in $GPUS; do
            local p="${pid_of[$g]:-}"
            if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then continue; fi
            local f; f=$(free_mb "$g")
            [ -z "$f" ] && continue
            if [ "$f" -ge "$FREE_MB" ]; then echo "$g"; return 0; fi
        done
        [ "$(date +%s)" -ge "$deadline" ] && return 1
        sleep "$POLL"
    done
}
for item in "${plan[@]}"; do
    s="${item%%:*}"; m="${item##*:}"
    if ! g=$(wait_for_slot); then
        log "TIMEOUT after ${MAX_WAIT_H}h waiting for ${FREE_MB} MiB free; stopping"
        log "re-run this script later, finished runs are skipped"
        exit 2
    fi
    d="benchmark_results_deeptherm/$TAG/random$s/member${m}_seed${m}"
    lf="$LOGS/random${s}_member${m}.log"
    t0=$(date +%s)
    log "start random$s member$m on GPU $g (free $(free_mb "$g") MiB)"
    ( export CUDA_VISIBLE_DEVICES="$g"
      nice -n "$NICE" python scripts/train.py --seed "$m" --member "$m" --split_seed "$s" \
          --csv "$CSV" --s_prior v2_exact --loss_weights 1 0 0 --monitor h \
          --tag_suffix _splitH --overwrite > "$lf" 2>&1 ) &
    pid_of[$g]=$!
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$s" "$m" "$m" "$g" running \
        "$(date -d @"$t0" '+%F %T')" "" "" >> "$STATUS"
    sleep 5
done
log "all ${#plan[@]} run(s) launched; waiting for the last ones to finish"
for g in $GPUS; do
    p="${pid_of[$g]:-}"
    [ -n "$p" ] && wait "$p" 2>/dev/null
done
wait
ok=0; bad=0
for item in "${plan[@]}"; do
    s="${item%%:*}"; m="${item##*:}"
    d="benchmark_results_deeptherm/$TAG/random$s/member${m}_seed${m}"
    if [ -f "$d/results.json" ]; then
        ok=$((ok+1)); log "done  random$s member$m"
    else
        bad=$((bad+1)); log "FAILED random$s member$m (see $LOGS/random${s}_member${m}.log)"
    fi
done
log "finished: $ok ok, $bad failed, wall clock $(( ($(date +%s) - start_all) / 60 )) min"
log "next: python check/divergence_report.py"
