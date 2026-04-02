export interface ChangelogEntry {
  id: string;
  version: string;
  date: string;
  title: string;
  highlights: ChangelogHighlight[];
  improvements?: string[];
  fixes?: string[];
  url: string;
}

export interface ChangelogHighlight {
  title: string;
  description: string;
}

/**
 * Changelog entries sourced from https://agpt.co/docs/platform/changelog/changelog
 *
 * To add a new entry:
 * 1. Add it to the top of the array
 * 2. Increment the version to match the platform version
 * 3. The popup will automatically show for users who haven't seen this version
 */
export const changelogEntries: ChangelogEntry[] = [
  {
    id: "2026-03-20",
    version: "v0.6.53",
    date: "March 20 – March 25, 2026",
    title: "Workflow Import, Dry-Run Mode & Marketplace Polish",
    highlights: [
      {
        title: "Import workflows from n8n, Make.com & Zapier",
        description:
          "Switching to AutoGPT no longer means starting from scratch. A new workflow import feature lets you bring in automations you've already built on other platforms and convert them into AutoGPT agents.",
      },
      {
        title: "Test your agents before they go live",
        description:
          "A new dry-run mode lets you simulate a full run of any agent without real-world side effects. Every block executes and produces realistic outputs, so you can verify your agent works correctly before it takes any real action.",
      },
      {
        title: "A more polished marketplace",
        description:
          "Card descriptions are now neatly truncated to keep the layout consistent, the download button has been repositioned for better flow, and card overflow issues have been resolved.",
      },
    ],
    improvements: [
      "Parallel AutoPilot actions — multiple steps now run simultaneously",
      "Scoped AutoPilot tools — control exactly which tools AutoPilot has access to",
      "Leaner tool schemas — 34% reduction in tool schema token cost",
      "Admin marketplace preview — preview and download agents before approving",
    ],
    fixes: [
      "Fixed blocks with complex inputs sometimes failing silently",
      "Fixed auto top-up setup showing a generic error with no payment method",
      "Fixed re-uploading a file to AutoPilot failing instead of replacing",
      "OAuth popup detection — app notices when you close an OAuth window",
      "Added circuit breaker to prevent infinite tool-call retry loops",
    ],
    url: "https://agpt.co/docs/platform/changelog/changelog/march-20-march-25-2026",
  },
  {
    id: "2026-03-13",
    version: "v0.6.52",
    date: "March 13 – March 20, 2026",
    title: "Platform Improvements & Bug Fixes",
    highlights: [
      {
        title: "Continued platform refinements",
        description:
          "This release focused on stability improvements, performance optimizations, and bug fixes across the platform.",
      },
    ],
    url: "https://agpt.co/docs/platform/changelog/changelog/march-13-march-20-2026",
  },
  {
    id: "2026-03-05",
    version: "v0.6.51",
    date: "March 5 – March 12, 2026",
    title: "Builder & Library Updates",
    highlights: [
      {
        title: "Builder and library experience improvements",
        description:
          "Enhancements to the agent building experience and library management features.",
      },
    ],
    url: "https://agpt.co/docs/platform/changelog/changelog/march-5-march-12-2026",
  },
];
