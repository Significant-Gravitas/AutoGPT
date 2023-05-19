import os
from datetime import datetime

from github import Github

from tests.integration.agent_factory import get_pr_review_agent
from tests.integration.challenges.utils import run_interaction_loop

PR_TARGET_BRANCH = "hackathon-pr-target"
PR_TARGET_REPO_USER = "merwanehamadi"
PR_TARGET_REPO_NAME = "Auto-GPT"
PR_TARGET_REPO = f"{PR_TARGET_REPO_USER}/{PR_TARGET_REPO_NAME}"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


def create_pr(source_branch_name, title, body):
    # First create a Github instance with your token:

    g = Github(GITHUB_TOKEN)

    # Then get your repository:
    repo = g.get_user(PR_TARGET_REPO_USER).get_repo(PR_TARGET_REPO_NAME)

    # Get the branch you want to copy

    base_branch = repo.get_branch(source_branch_name)

    # Create the name for the new branch
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_branch_name = f"{source_branch_name}-{timestamp}"

    # Create the new branch
    repo.create_git_ref(ref=f"refs/heads/{new_branch_name}", sha=base_branch.commit.sha)

    # Create a new pull request
    pr = repo.create_pull(
        title=title,
        body=body,
        head=new_branch_name,
        base=PR_TARGET_BRANCH,
    )
    return pr.number


def check_pr(pr_number, parameters):
    # First create a Github instance with your token:
    g = Github(GITHUB_TOKEN)

    # Get the repository
    repo = g.get_user(PR_TARGET_REPO_USER).get_repo(PR_TARGET_REPO_NAME)

    # Get the pull request
    pr = repo.get_pull(pr_number)

    # Count approvals
    approvals = 0

    # Get reviews for the pull request
    for review in pr.get_reviews():
        # Check if the review is an approval
        if review.state == "APPROVED":
            approvals += 1

    print(
        f"The PR number {pr_number} in the repository {PR_TARGET_REPO} has {approvals} approvals."
    )
    if parameters.approved:
        assert approvals > 0
    else:
        assert approvals == 0


def run_tests(parameters, monkeypatch, workspace):
    pr_number = create_pr(
        parameters.source_branch_name, parameters.title, parameters.body
    )
    review_agent = get_pr_review_agent(pr_number, PR_TARGET_REPO, workspace)
    # run_interaction_loop(monkeypatch, review_agent, parameters.cycle_count)
    check_pr(pr_number, parameters)
