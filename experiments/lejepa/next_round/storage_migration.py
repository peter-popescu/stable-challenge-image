"""Verified, bounded migration of inactive external evaluation artifacts to NBU.

No recursive deletion: each home file is hashed against its verified scratch copy
immediately before unlink; only empty recorded directories are then removed.
"""
import argparse,hashlib,json,os,re,shutil,stat,subprocess,time
from pathlib import Path
from ..sweep import atomic_json

SOURCE=Path('experiment_outputs/next-round/external')
SCRATCH=Path('/nbu/people/ppopescu/stable-challenge-image-next-round')
AUDIT=Path('experiment_outputs/next-round/reporting')
UID=os.getuid()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def jobs_snapshot(label,audit):
    ids=subprocess.check_output(['squeue','-h','-u',str(UID),'-o','%i'],text=True).split()
    rows=[]
    for job in ids:
        text=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
        acct=subprocess.check_output(['sacct','-j',job,'-X','-n','-P','--format=JobID,State,SubmitLine%4000'],text=True)
        state=re.search(r'\bJobState=(\S+)',text).group(1)
        row=dict(job_id=job,state=state,scontrol=text,submit_line=acct)
        rows.append(row)
        # Ignore this migration itself. The unrelated interactive shell is only
        # inspected; neither it nor any training allocation is modified.
        active=state in ('RUNNING','COMPLETING','CONFIGURING')
        signals=('next_round.external','external-extract','external-cpu','external-new',str(SOURCE),str(SCRATCH/'external'))
        if active and job!=os.environ['SLURM_JOB_ID'] and any(s in acct+text for s in signals):
            atomic_json(audit/('jobs-'+label+'.json'),rows)
            raise RuntimeError('Active external consumer detected: '+job)
    atomic_json(audit/('jobs-'+label+'.json'),rows)


def check_owned(path,directory=False):
    s=path.lstat()
    if s.st_uid!=UID or stat.S_ISLNK(s.st_mode):raise RuntimeError('Unexpected owner/symlink: '+str(path))
    if directory and not stat.S_ISDIR(s.st_mode):raise RuntimeError('Not a directory: '+str(path))
    return s


def inventory(root):
    check_owned(root,True);files={};dirs=[]
    for directory,children,names in os.walk(root,followlinks=False):
        directory=Path(directory);check_owned(directory,True)
        dirs.append(str(directory.relative_to(root)))
        for name in children:check_owned(directory/name,True)
        for name in names:
            path=directory/name;s=check_owned(path)
            if not stat.S_ISREG(s.st_mode):raise RuntimeError('Nonregular file: '+str(path))
            key=str(path.relative_to(root))
            files[key]=dict(size=s.st_size,sha256=sha(path),mtime_ns=s.st_mtime_ns,mode=stat.S_IMODE(s.st_mode),uid=s.st_uid,gid=s.st_gid)
    return dict(files=files,directories=sorted(dirs))


def compare(expected,actual,label):
    if expected!=actual:raise RuntimeError('Inventory/content/metadata mismatch: '+label)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--resume-staging',type=Path);args=parser.parse_args()
    job=os.environ.get('SLURM_JOB_ID')
    if not job:raise RuntimeError('Slurm required')
    audit=AUDIT/('storage-migration-'+job);audit.mkdir(parents=True,exist_ok=False)
    source=SOURCE.absolute();dest=SCRATCH/'external';future=SOURCE.with_name('external-new').absolute()
    backup=source.with_name('external.verified-home-copy-'+job)
    staging=args.resume_staging if args.resume_staging else SCRATCH/('external.staging-'+job)
    if args.resume_staging:
        if staging.parent.resolve()!=SCRATCH.resolve() or not re.fullmatch(r'external\.staging-\d+',staging.name):raise RuntimeError('Resume path is not a bounded task staging directory')
        check_owned(staging,True)
    status=dict(job_id=job,host=os.uname().nodename,source=str(source),destination=str(dest),backup=str(backup),staging=str(staging),phase='preflight',started=time.time())
    def update(phase,**extra):
        status.update(phase=phase,**extra);atomic_json(audit/'status.json',status);print(phase,flush=True)
    update('preflight')
    jobs_snapshot('initial',audit)
    check_owned(source,True);check_owned(SCRATCH.parent,True)
    if source.is_symlink() or dest.exists() or (staging.exists() and not args.resume_staging) or backup.exists() or future.exists() or future.is_symlink():
        raise RuntimeError('Migration paths are not in expected initial state')
    if SCRATCH.exists():check_owned(SCRATCH,True)
    else:SCRATCH.mkdir(mode=0o750)
    disk=shutil.disk_usage(SCRATCH)
    if disk.free<100*1024**3:raise RuntimeError('Scratch has less than100 GiB available')
    probe=SCRATCH/('.slurm-write-read-'+job)
    payload=os.urandom(256)
    with probe.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    check_owned(probe)
    if probe.read_bytes()!=payload:raise RuntimeError('Compute-node scratch write/read failed')
    probe.unlink()
    update('inventory',scratch_free_bytes=disk.free,compute_node_write_read_verified=True)
    original=inventory(source);atomic_json(audit/'inventory.json',original)
    update('copying',file_count=len(original['files']),total_bytes=sum(r['size'] for r in original['files'].values()))
    resumed=dict(reused_files=0,reused_bytes=0,copied_files=0,copied_bytes=0)
    if args.resume_staging:
        for d,children,names in os.walk(staging,followlinks=False):
            check_owned(Path(d),True)
            for child in children:check_owned(Path(d)/child,True)
            for name in names:
                st=check_owned(Path(d)/name)
                if not stat.S_ISREG(st.st_mode):raise RuntimeError('Nonregular resumed staging file')
    def copy_or_reuse(src,dst):
        src=Path(src);dst=Path(dst)
        if dst.exists():
            check_owned(dst)
            if dst.stat().st_size==src.stat().st_size and sha(dst)==original['files'][str(src.relative_to(source))]['sha256']:
                shutil.copystat(src,dst)
                resumed['reused_files']+=1;resumed['reused_bytes']+=src.stat().st_size
                return str(dst)
        shutil.copy2(src,dst)
        resumed['copied_files']+=1;resumed['copied_bytes']+=src.stat().st_size
        return str(dst)
    shutil.copytree(source,staging,copy_function=copy_or_reuse,dirs_exist_ok=bool(args.resume_staging))
    update('copied-awaiting-verification',resume_statistics=resumed)
    copied=inventory(staging);compare(original,copied,'scratch copy')
    atomic_json(audit/'scratch-inventory.json',copied)
    compare(original,inventory(source),'source stability after copy')
    jobs_snapshot('before-publish',audit)
    update('verified-before-publish')
    # Bounded renames publish the verified directory without rewriting cache paths.
    staging.rename(dest)
    source.rename(backup)
    temporary=source.with_name('external.link-'+job)
    temporary.symlink_to(dest,target_is_directory=True)
    temporary.replace(source)
    newdest=SCRATCH/'external-new';newdest.mkdir(mode=0o750,exist_ok=False)
    future.symlink_to(newdest,target_is_directory=True)
    if source.resolve()!=dest.resolve() or future.resolve()!=newdest.resolve():raise RuntimeError('Published links did not resolve correctly')
    jobs_snapshot('before-home-cleanup',audit)
    update('published-and-verified',future_external_path=str(future),future_scratch_path=str(newdest))
    # Only exact verified duplicates are unlinked. No rm/rmtree or wildcard cleanup.
    with (audit/'removed-files.jsonl').open('x') as journal:
        for index,(relative,record) in enumerate(original['files'].items()):
            old=backup/relative;new=dest/relative
            a=check_owned(old);b=check_owned(new)
            if a.st_size!=record['size'] or b.st_size!=record['size'] or sha(old)!=record['sha256'] or sha(new)!=record['sha256']:
                raise RuntimeError('Refusing to remove changed/unverified duplicate: '+relative)
            old.unlink()
            journal.write(json.dumps(dict(path=relative,source_sha256=record['sha256'],verified_destination_sha256=record['sha256'],size_bytes=record['size'],removed=True))+'\n')
            if index%200==0:journal.flush()
        journal.flush();os.fsync(journal.fileno())
    for relative in sorted(original['directories'],key=lambda x:len(Path(x).parts),reverse=True):
        path=backup/relative;check_owned(path,True);path.rmdir()
    compare(original,inventory(dest),'final scratch contents')
    update('complete',completed=time.time(),home_duplicates_removed=True,all_files_sha256_verified=True)
    atomic_json(AUDIT/'external-storage.json',status)
    print('READY',str(dest),flush=True)

if __name__=='__main__':main()
