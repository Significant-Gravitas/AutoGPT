jc git log --all --stat | jq > all.json
jq -c -r  ".[]?|[.author_email,.stats.files[]?]" all.json |sort |uniq -c | sort -n > report2.txt
