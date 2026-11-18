#!/usr/bin/env bash
# Approved byte transfer and a success-only scheduler release; no image work.
set -euo pipefail
dataset="${1:?dataset}"
prepare_job="${2:?held preparation job ID}"
if [ "$dataset" != clevr ]; then
  printf '%s\n' 'This scratch transfer/release wrapper is scoped to CLEVR' >&2
  exit 2
fi
python3 - <<'CHECK'
from pathlib import Path
actual=Path('experiment_outputs/next-round/external').resolve()
allowed=Path('/nbu/people/ppopescu/stable-challenge-image-next-round/external').resolve()
if actual!=allowed:
    raise SystemExit('CLEVR requires the verified external scratch symlink before transfer')
CHECK
bash experiments/lejepa/next_round/external_transfer.sh "$dataset"
python3 - "$dataset" "$prepare_job" <<'RELEASE'
import json,subprocess,sys,time
from pathlib import Path
root=Path('experiment_outputs/next-round/external')
p=root/(sys.argv[1]+'-jobs.json');record=json.loads(p.read_text())
if record['jobs']['prepare']['job_id']!=sys.argv[2]:raise SystemExit('Preparation job identity mismatch')
archive=root/'data/clevr/CLEVR_v1.0.zip'
if not archive.is_file() or archive.stat().st_size<=0:raise SystemExit('No completed archive; preparation remains held')
# Atomic final archive publication occurs only after curl succeeds. ZIP CRCs,
# archive SHA and all decompression are deliberately performed in Slurm later.
subprocess.run(['scontrol','release',sys.argv[2]],check=True)
record['jobs']['prepare']['held']=False
record['transfer']=dict(status='complete',archive=str(archive),bytes=archive.stat().st_size,
    release_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    verification='ZIP integrity/SHA256/extraction in released Slurm preparation')
tmp=p.with_suffix('.json.transfer.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p)
print(json.dumps(dict(archive=str(archive),bytes=record['transfer']['bytes'],released_prepare_job=sys.argv[2])),flush=True)
RELEASE
