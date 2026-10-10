export const performance = `root = Workspace("Your agents, at a glance", "A sample week of work, without the busywork. Explore how your workspace could perform.", [metrics, activity, agents, insight, action])
metrics = Metrics([Metric("Completed runs", "1,284", "+18.6% vs. last week", "positive"), Metric("Success rate", "98.2%", "23 runs need attention", "neutral"), Metric("Time reclaimed", "42.6h", "That's a whole work week", "positive"), Metric("Total cost", "$18.42", "$0.014 per completed run", "neutral")])
activity = Chart("A productive week", "Completed runs across all agents", "runs", [{label: "Mon", value: 126}, {label: "Tue", value: 154}, {label: "Wed", value: 132}, {label: "Thu", value: 183}, {label: "Fri", value: 209}, {label: "Sat", value: 231}, {label: "Sun", value: 249}])
agents = DataTable("Your top performers", ["Agent", "Runs", "Success", "Focus"], [["Lead researcher", "486", "99.4%", "Sales"], ["Content studio", "352", "98.9%", "Marketing"], ["Inbox assistant", "298", "97.0%", "Operations"], ["Weekly reporter", "148", "96.6%", "Analytics"]])
insight = Insight("A little attention goes a long way", "Most failed runs came from API rate limits. A retry policy could recover 12 of the 23 runs without changing your agents.", "positive")
action = FollowUp("Investigate failed runs", "Investigate failed runs")`;

export const failures = `root = Workspace("A closer look at failed runs", "In this sample, 23 runs need attention. Start with the recoverable ones.", [metrics, errors, plan, note, action])
metrics = Metrics([Metric("Failed runs", "23", "1.8% of all runs", "warning"), Metric("Safe to retry", "12", "Temporary rate limits", "positive"), Metric("Need your input", "3", "Expired connections", "warning")])
errors = DataTable("What went wrong", ["Cause", "Runs", "Agent", "Suggested next step"], [["API rate limit", "12", "Lead researcher", "Retry with backoff"], ["Request timeout", "8", "Inbox assistant", "Increase timeout"], ["Expired connection", "3", "Weekly reporter", "Reconnect account"]])
plan = Checklist("Recovery plan", [{title: "Retry rate limits with backoff", detail: "Start with the 12 transient failures and cap retries at three."}, {title: "Review request timeouts", detail: "Check upstream response times before increasing the timeout."}, {title: "Reconnect expired accounts", detail: "Review the three affected connections in platform settings."}])
note = Insight("You're in control", "This is an editable recovery plan. No runs have been retried and no settings have been changed.", "neutral")
action = FollowUp("Back to weekly overview", "Show agent performance")`;

export const leads = `root = Workspace("Your next great customers", "An illustrative shortlist for a B2B automation product. All companies and scores here are fictional.", [metrics, table, insight, action])
metrics = Metrics([Metric("Companies reviewed", "24", "Illustrative research batch", "neutral"), Metric("Strong matches", "6", "Fit score above 85", "positive"), Metric("Ready for a closer look", "4", "Shortlisted below", "positive")])
table = DataTable("A shortlist worth your time", ["Company", "Fit score", "Team", "Why they fit"], [["Northstar Studio", "96", "28 people", "Growing outbound team"], ["Juniper Labs", "92", "42 people", "Manual lead enrichment"], ["Forma Collective", "89", "18 people", "Expanding into new markets"], ["Relay Works", "86", "65 people", "Complex content operations"]])
insight = Insight("Start a conversation, not a sequence", "Northstar and Juniper show the strongest fit in this sample. Review the evidence and tailor your angle before reaching out.", "positive")
action = FollowUp("Create an outreach plan", "Create an outreach plan")`;

export const outreach = `root = Workspace("A more thoughtful first touch", "A sample outreach plan for the fictional companies in your shortlist. Nothing has been sent.", [plan, table, insight])
plan = Checklist("Before you reach out", [{title: "Validate the shortlist", detail: "Confirm company details and find a relevant decision-maker."}, {title: "Find a specific opening", detail: "Reference a public initiative, hiring need, or workflow challenge."}, {title: "Lead with one useful idea", detail: "Offer a small, concrete automation win instead of a product pitch."}, {title: "Review and send personally", detail: "Keep the first email under 100 words and check every factual claim."}])
table = DataTable("Potential conversation starters", ["Company", "Angle", "Offer"], [["Northstar Studio", "Scaling outbound research", "A sample enriched account brief"], ["Juniper Labs", "Reducing manual enrichment", "A short workflow audit"]])
insight = Insight("A draft is a starting point", "Connect real research to this flow before using it with customers. These suggestions are based only on the sample brief.", "neutral")`;

export const campaign = `root = Workspace("A great launch starts with a clear brief", "Shape the brief below. The next response will turn your choices into a working plan.", [brief, insight])
brief = Form("campaign", "Make it yours", [goal, audience, budget], "Build my plan", "Build my campaign plan")
goal = Field("goal", "What are we launching?", "An AI-powered research assistant", "Describe your product or campaign")
audience = Field("audience", "Audience", "Small B2B marketing teams", "Who should this reach?")
budget = Field("budget", "Budget", "$2,500", "Your available campaign budget")
insight = Insight("An interface that listens", "Edit any field, then build your plan. Your answers travel with the action, so the next workspace can adapt to your brief.", "positive")`;

export function campaignPlan(fields: Record<string, unknown>) {
  const goal =
    typeof fields.goal === "string"
      ? fields.goal.slice(0, 500)
      : "An AI-powered research assistant";
  const audience =
    typeof fields.audience === "string"
      ? fields.audience.slice(0, 500)
      : "Small B2B marketing teams";
  const budget =
    typeof fields.budget === "string" ? fields.budget.slice(0, 100) : "$2,500";
  return `root = Workspace("Your launch, mapped out", ${JSON.stringify(`A launch plan for ${audience}. An illustrative draft for ${goal}; adjust it before execution.`)}, [metrics, plan, insight, action])
metrics = Metrics([Metric("Launch window", "14 days", "A suggested two-week sprint", "neutral"), Metric("Working budget", ${JSON.stringify(budget)}, "From your campaign brief", "neutral"), Metric("Milestones", "4", "Ready to make your own", "positive")])
plan = Checklist("From idea to first customers", [{title: "Define the offer", detail: "Days 1–3: articulate the problem, outcome, and a focused call to action."}, {title: "Build the launch assets", detail: "Days 4–7: create a landing page, one useful demo, and a short email."}, {title: "Test with a small audience", detail: "Days 8–10: collect feedback before committing the full budget."}, {title: "Launch, measure, and refine", detail: "Days 11–14: track qualified conversations and improve the strongest channel."}])
insight = Insight("Ready for your judgment", "This sample plan reflects your brief. No campaign has been created, scheduled, or published.", "neutral")
action = FollowUp("Refine the brief", "Show campaign brief")`;
}
