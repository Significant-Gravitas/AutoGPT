from collections import defaultdict
import click
from datetime import datetime, timezone

from github import Auth, Github
from github.NamedUser import NamedUser
from github.PaginatedList import PaginatedList
from github.PullRequest import PullRequest
from github.PullRequestComment import PullRequestComment

from .model import LatencyStats, PullRequestMetrics


@click.command()
@click.argument("repo")
def main(repo: str):
    import os

    from dotenv import load_dotenv

    print(f"Analyzing repo {repo}")

    # Clean repo name
    repo = "/".join(
        repo.rsplit(":", 1)[-1].rsplit("/", 2)[-2:]
    ).removesuffix(".git")
    print(f"Cleaned repo name: {repo}")

    # Set up GitHub
    load_dotenv()
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        raise ValueError("GITHUB_TOKEN is not set")

    github = Github(auth=Auth.Token(github_token))
    github_repo = github.get_repo(repo)
    print(f"Repo description: {github_repo.description}")

    maintainers = github_repo.get_collaborators(permission="maintain")
    print("Maintainers:")
    for maintainer in maintainers:
        print(f"- @{maintainer.login}")
    print()

    open_pulls = github_repo.get_pulls("open")
    print(f"Repo has {open_pulls.totalCount} open PRs")
    for pr in open_pulls:
        print()
        metrics = get_pr_metrics(maintainers, pr)

        print(metrics.model_dump_json(indent=4))

        if metrics.days_since_maintainer_comment is None:
            print(
                f"⚠️ NO MAINTAINER COMMENT AFTER {metrics.days_since_opened} DAYS!!!"
            )


def get_pr_metrics(
    maintainers: PaginatedList[NamedUser], pr: PullRequest
) -> PullRequestMetrics:
    print(f"PR #{pr.number} - {pr.title}")
    print(pr.html_url)

    print(
        f"Author: @{pr.user.login}"
        + (" (maintainer)" if pr.user in maintainers else "")
    )
    metrics = PullRequestMetrics(
        days_since_opened=days_since(pr.created_at),
        days_since_author_comment=days_since(pr.created_at),
        days_since_maintainer_comment=None,
        days_since_maintainer_review=None,
        days_until_first_maintainer_comment=None,
        days_until_first_maintainer_review=None,
        author_response_stats=None,
        maintainer_response_stats=None,
    )

    print(f"Opened: {pr.created_at} ({metrics.days_since_opened} days ago)")

    if pr.updated_at:
        days_since_activity = days_since(pr.updated_at)
        print(f"Updated: {pr.updated_at} ({days_since_activity} days ago)")

    # from pprint import pprint
    # print("Events:")
    # review_request_events = []
    # for event in pr.get_issue_events():
    #     print(f"{event.created_at} <@{event.actor.login}> {event.event}")
    #     pprint(vars(event._rawData))
    #     if event.event == "review_requested":
    #         review_request_events.append(event)

    # Reviews
    review_threads: dict[int, list[PullRequestComment]] = defaultdict(list)
    if (reviews := pr.get_reviews()).totalCount > 0:
        for review in reviews:
            print(f"🪵 Review {review.id} - {review.submitted_at} "
                  f"({days_between(pr.created_at, review.submitted_at)} days since PR creation)")
            print(f'"{review.body}"')

            # FIXME: Measure response time relative to latest review_requested, ready_for_review, or commit
            if review.user in maintainers and review.user != pr.user:
                metrics.days_until_first_maintainer_review = min(
                    _interval := days_between(pr.created_at, review.submitted_at),
                    metrics.days_until_first_maintainer_review or _interval,
                )
                metrics.days_since_maintainer_review = min(
                    _interval := days_since(review.submitted_at),
                    metrics.days_since_maintainer_review or _interval,
                )
                if metrics.maintainer_response_stats is None:
                    metrics.maintainer_response_stats = LatencyStats(series=[])
                metrics.maintainer_response_stats.series.append(
                    _interval := days_between(pr.created_at, review.submitted_at)
                )
                print(f"➡️ Adding {_interval} day interval to maintainer stats")

            # Review comments
            for review_comment in pr.get_single_review_comments(review.id):
                thread_id = review_comment.in_reply_to_id or review_comment.id
                thread = review_threads[thread_id]

                if review_comment.user == pr.user:
                    print(
                        f"🪵 Author responded: {review_comment.created_at} "
                        f"({days_since(review_comment.created_at)} days ago)"
                    )
                    print(
                        f"🪵 ID: {review_comment.id}; REPLY TO: {review_comment.in_reply_to_id}"
                    )
                    print(f'🪵 "{review_comment.body}"\n')

                    metrics.days_since_author_comment = min(
                        days_since(review_comment.created_at),
                        metrics.days_since_author_comment,
                    )

                    if thread and (last_comment := thread[-1]).user in maintainers:
                        if metrics.author_response_stats is None:
                            metrics.author_response_stats = LatencyStats(series=[])
                        metrics.author_response_stats.series.append(
                            # HACK: assuming the user comment is a response to the last maintainer comment
                            _interval := days_between(
                                last_comment.created_at, review_comment.created_at
                            )
                        )
                        print(
                            f"➡️ Adding {_interval} day interval to "
                            f"author stats ({review_comment.id} <> {last_comment.id})"
                        )

                elif review_comment.user in maintainers:
                    print(
                        f"🪵 Maintainer reviewed: {review_comment.created_at} "
                        f"({days_since(review_comment.created_at)} days ago; "
                        f"{days_between(pr.created_at, review_comment.created_at)} days after opening)"
                    )
                    print(
                        f"🪵 ID: {review_comment.id}; REPLY TO: {review_comment.in_reply_to_id}"
                    )
                    print(f'🪵 "{review_comment.body}"\n')
                    metrics.days_since_maintainer_comment = min(
                        _interval := days_since(review_comment.created_at),
                        metrics.days_since_maintainer_comment or _interval,
                    )
                    metrics.days_until_first_maintainer_comment = min(
                        _interval := days_between(pr.created_at, review_comment.created_at),
                        metrics.days_until_first_maintainer_comment or _interval,
                    )

                    last_comment = None
                    if thread and (last_comment := thread[-1]).user not in maintainers:
                        if metrics.maintainer_response_stats is None:
                            metrics.maintainer_response_stats = LatencyStats(series=[])
                        metrics.maintainer_response_stats.series.append(
                            _interval := days_between(
                                (
                                    last_comment
                                    if last_comment and last_comment.user == pr.user
                                    else pr
                                ).created_at,
                                review_comment.created_at,
                            )
                        )
                        print(
                            f"➡️ Adding {_interval} day interval to "
                            f"maintainer stats ({review_comment.id} <> "
                            f"{last_comment.id if last_comment and last_comment.user == pr.user else f'PR {pr.number}'})"
                        )
                else:
                    continue

                thread.append(review_comment)

    # Regular comments
    if pr.comments:
        last_comment = None

        for comment in pr.get_issue_comments():
            if comment.user == pr.user:
                print(
                    f"🪵 Author commented: {comment.created_at} "
                    f"({days_since(comment.created_at)} days ago)"
                )
                print(f'🪵 "{comment.body}"\n')
                metrics.days_since_author_comment = min(
                    days_since(comment.created_at),
                    metrics.days_since_author_comment,
                )

                if last_comment and last_comment.user in maintainers:
                    if metrics.author_response_stats is None:
                        metrics.author_response_stats = LatencyStats(series=[])
                    metrics.author_response_stats.series.append(
                        # HACK: assuming the user comment is a response to the last maintainer comment
                        _interval := days_between(last_comment.created_at, comment.created_at)
                    )
                    print(
                        f"➡️ Adding {_interval} day interval to "
                        f"author stats ({comment.id} <> {last_comment.id})"
                    )
                last_comment = comment

            elif comment.user in maintainers:
                print(
                    f"🪵 Maintainer commented: {comment.created_at} "
                    f"({days_since(comment.created_at)} days ago; "
                    f"{days_between(pr.created_at, comment.created_at)} days after opening)"
                )
                print(f'🪵 "{comment.body}"\n')
                metrics.days_since_maintainer_comment = min(
                    _interval := days_since(comment.created_at),
                    metrics.days_since_maintainer_comment or _interval,
                )
                metrics.days_until_first_maintainer_comment = min(
                    _interval := days_between(pr.created_at, comment.created_at),
                    metrics.days_until_first_maintainer_comment or _interval,
                )

                if not last_comment or last_comment.user not in maintainers:
                    if metrics.maintainer_response_stats is None:
                        metrics.maintainer_response_stats = LatencyStats(series=[])
                    metrics.maintainer_response_stats.series.append(
                        _interval := days_between(
                            (
                                last_comment
                                if last_comment and last_comment.user == pr.user
                                else pr
                            ).created_at,
                            comment.created_at,
                        )
                    )
                    print(
                        f"➡️ Adding {_interval} day interval to "
                        f"maintainer stats ({comment.id} <> "
                        f"{last_comment.id if last_comment and last_comment.user == pr.user else f'PR {pr.number}'})"
                    )
                last_comment = comment

    return metrics


def days_since(since: datetime) -> int:
    return (datetime.now(timezone.utc) - since).days


def days_between(first_event: datetime, last_event: datetime) -> int:
    return (last_event - first_event).days


if __name__ == "__main__":
    main()
