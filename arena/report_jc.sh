jc git log --all --stat --since 2023-09-01 | jq > all_30.json
jq -c -r  ".[]?|[.author_email,.stats.files[]?]" all.json |sort |uniq -c | sort -n > report2.txt
jq -c -r  ".[]?|.author_email" all.json |sort |uniq -c | sort -n > report4.txt
