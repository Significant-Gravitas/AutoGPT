# create remote for all json
for x in `jq -r .github_repo_url *.json`; do git remote add $(echo $x| cut -d/ -f4 ) $x; done
# fetch tehm
for x in `git remote`; do git fetch $x; done
# log them
git log --all --patch >all.txt

# author repotr
grep Author: all.txt  | sort |uniq -c |sort -n > author_stats.txt

# file report
grep "diff --git" all.txt  |sort | uniq -c | sort -n > files.txt
