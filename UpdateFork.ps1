# PowerShell script to update a forked Git repository

# Navigate to your local repository
# Replace this with the path to your local repository
$localRepoPath = $PSScriptRoot

# URL of the original repository (upstream)
# Replace this with the URL of the original repository
$upstreamRepoUrl = "https://github.com/Significant-Gravitas/AutoGPT.git"

# Name of your branch to update (usually master or main)
$branchName = "master"

# Change to the repository directory
Set-Location -Path $localRepoPath

# Add the original repository as a remote called 'upstream' if it doesn't already exist
$upstreamExists = git remote | Select-String "upstream" -Quiet
if (-not $upstreamExists) {
    git remote add upstream $upstreamRepoUrl
}

# Fetch the latest changes from the original repository
git fetch upstream

# Checkout your branch
git checkout $branchName

# Merge the changes from the upstream repository
git merge upstream/$branchName

# Push the updates to your forked repository
git push origin $branchName
