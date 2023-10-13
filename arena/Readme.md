
# most common edited files

```
python report.py all.json 
```

That produces user_files.csv

```
cut -d, -f3 user_files.csv  |sort |uniq -c | sort -n > mostcommon.txt

```
