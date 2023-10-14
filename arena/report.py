# report on the jc output
import click
import json
from collections import Counter
import pandas as pd

@click.command()
@click.argument('infile', type=click.File('r'))
def main(infile):
    #user_repo = {}
    user_repo2 = {}
    user_files = []
    df = json.load(infile)
    for commit in df:
        if "stats" in commit:
            #print(commit)
            for stat in commit["stats"]:
                for filen in commit["stats"]["files"]:
                    #user_repo[filen] = commit["author_email"]
                    if "arena/" in filen:
                        user_repo2[commit["author_email"]] = filen
                        filen = "<ARENA>"
                    if "benchmark/" in filen:
                        filen = "<benchmark>"
                    if ".png" in filen:
                        filen = "<image>"
                    if ".jpg" in filen:
                        filen = "<image>"
                    if ".md" in filen:
                        filen = "<markdown>"
                    if "/reports/" in filen:
                        filen = "<reports>"
                    if ".json" in filen:
                        filen = "<reports>"
                    if ".log" in filen:
                        filen = "<reports>"
                    for name in ["challenges/",""]:
                        if "challenges/" in filen:
                            filen = "<challenge>"
                    
                    for name in [".env",
                                 "pyproject.toml",
                                 "poetry.lock",
                                 "requirements.txt",
                                 "Docker",".gitmodules","docker-compose.yml"]:
                        if name in filen:
                            filen = "<infra>"
                    if "/combined_charts/" in filen:
                        filen = "<charts>"
                    #print (filen,commit["author_email"])
                    user_files.append(dict(filename=filen,name=commit["author_email"]))
    ud = pd.DataFrame(user_files)
    #import pdb
    #pdb.set_trace()
    ud2= ud.groupby(["name"])["filename"].unique().apply('|'.join).reset_index(name="edited_files")
    #print("DEBUG",ud2)
    #ud2.to_csv("user_files.csv")

    #result = df.groupby('username')['filename'].unique().apply(', '.join).reset_index(name='edited_files')
    #ixsmport pdb
    #opdb.set_trace()
# Filter users with only one contribution
    filtered_result = ud2[ (ud2["edited_files"].str.count("\|") >0) & (ud2["edited_files"].str.contains("ARENA")) ] 
    df2 = pd.DataFrame.from_dict(user_repo2,orient="index")
    #import pdb
    #pdb.set_trace()
    df2.index.name='name'
    filtered_result2 =pd.merge(filtered_result,df2,on="name",how="inner").drop(columns=["name"])
    #filtered_result2.rename(columns={'' : "name" })
    filtered_result2.rename({'index':'word',0:'name'},axis='columns',inplace=True)
    
    #filtered_result2.rename("0").name='name'

    
    filtered_result2.insert(0,"name",filtered_result2.pop("name"))
    filtered_result2.to_csv("filtered_result2.csv")    
    #for row in df:
    #    print(row)
    # now for each file lets rank them
    filtered_result2["tokens"] = filtered_result2["edited_files"].str.split("\|")

    all_tokens = [token for sublist in filtered_result2['tokens'] for token in sublist]
    token_counts = Counter(all_tokens)
    file_counts = pd.DataFrame(token_counts.most_common())

    file_counts.rename({'index':'word',0:'file'},axis='columns',inplace=True)
    file_counts.rename({'index':'word',1:'count'},axis='columns',inplace=True)
    file_counts[file_counts["count"]>1].to_csv("TokenCounts.csv")
        
if __name__ =="__main__":
    main()
