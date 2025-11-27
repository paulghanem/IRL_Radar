#!/bin/bash

# Get list of running jobs
squeue -u pghanem -o "%A" | tail -n +2 > /tmp/running_jobs.txt

count=0
# Delete output and error files from non-running jobs
for file in *.out *.err; do
  if [ -f "$file" ]; then
    jobid=$(echo "$file" | grep -oE '[0-9]{8}' | head -1)
    if [ -n "$jobid" ] && ! grep -q "^$jobid$" /tmp/running_jobs.txt; then
      rm -f "$file"
      ((count++))
    fi
  fi
done

echo "Deleted $count old output/error files"
ls -1 *.out *.err 2>/dev/null | wc -l
echo "files remaining"
