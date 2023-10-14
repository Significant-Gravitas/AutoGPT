jc git log --all --stat | jq > all.json
python report.py ./all.json 
