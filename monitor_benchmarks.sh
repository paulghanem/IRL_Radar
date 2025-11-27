#!/bin/bash
JOBS="36233168 36233169 36233170 36233171"
echo "Monitoring benchmark jobs..."
echo ""

while true; do
  ALL_DONE=true
  for job in $JOBS; do
    STATUS=$(squeue -u $USER -j $job 2>/dev/null | tail -1 | awk '{print $5}')
    if [ -n "$STATUS" ]; then
      ALL_DONE=false
      echo "Job $job: $STATUS"
    fi
  done
  
  if [ "$ALL_DONE" = true ]; then
    echo ""
    echo "All jobs completed! Checking results..."
    sacct -j 36233168,36233169,36233170,36233171 --format=JobID,State,ExitCode,Elapsed
    break
  fi
  
  echo "Waiting... ($(date +%H:%M:%S))"
  sleep 30
done
