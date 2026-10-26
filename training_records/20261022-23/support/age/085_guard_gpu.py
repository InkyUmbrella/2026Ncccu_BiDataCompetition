import subprocess,sys
physical=sys.argv[1]
apps=subprocess.run(['nvidia-smi','--id='+physical,'--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
if apps: raise SystemExit('GPU '+physical+' already has compute processes: '+apps)
used=int(subprocess.run(['nvidia-smi','--id='+physical,'--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip())
if used>512: raise SystemExit('GPU '+physical+' has unexpected memory usage: '+str(used))
print('GPU '+physical+' available; memory MiB='+str(used),flush=True)
