# report on the jc output
import click
import json
import pandas as pd

@click.command()
@click.argument('infile', type=click.File('r'))
def main(infile):
    user_repo = {}
    user_repo2 = {}
    user_files = []
    df = json.load(infile)
    for commit in df:
        if "stats" in commit:
            #print(commit)
            for stat in commit["stats"]:
                for filen in commit["stats"]["files"]:
                    user_repo[filen] = commit["author_email"]
                    user_repo2[commit["author_email"]] = filen                    
                    if "arena/" in filen:
                        filen = "<ARENA>"
                    #print (filen,commit["author_email"])
                    user_files.append(dict(filename=filen,name=commit["author_email"]))
    ud = pd.DataFrame(user_files)
    #import pdb
    #pdb.set_trace()
    ud2= ud.groupby(["name"])["filename"].unique().apply('|'.join).reset_index(name="edited_files")
    #print("DEBUG",ud2)
    ud2.to_csv("user_files.csv")

    #result = df.groupby('username')['filename'].unique().apply(', '.join).reset_index(name='edited_files')
    #ixsmport pdb
    #opdb.set_trace()
# Filter users with only one contribution
    filtered_result = ud2[ (ud2["edited_files"].str.count("\|") >1) & (ud2["edited_files"].str.contains("ARENA")) ] 
    df2 = pd.DataFrame.from_dict(user_repo2,orient="index")
    import pdb
    pdb.set_trace()
    
    filtered_result2 =pd.merge(filtered_result,df2,on="name",how="inner")
    
    filtered_result2.to_csv("filtered_result2.csv")

    #for row in df:
    #    print(row)
    
    
if __name__ =="__main__":
    main()
