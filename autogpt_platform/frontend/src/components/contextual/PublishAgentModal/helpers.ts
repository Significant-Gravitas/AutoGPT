import type { GraphModel } from "@/app/api/__generated__/models/graphModel";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { LiveStoreSubmission } from "@/app/api/__generated__/models/liveStoreSubmission";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import type { PublishAgentInfoInitialData } from "./components/AgentInfoStep/helpers";

export const LISTING_CHECK_FAILED_MESSAGE =
  "We couldn't check whether this workflow is already in the Marketplace.";

export const emptyModalState = {
  agent_id: "",
  title: "",
  subheader: "",
  slug: "",
  thumbnailSrc: "",
  youtubeLink: "",
  category: "",
  description: "",
  recommendedScheduleCron: "",
  instructions: "",
  agentOutputDemo: "",
  changesSummary: "",
  additionalImages: [],
};

export interface AgentFormSource {
  name: string;
  description: string;
  imageSrc: string;
  recommendedScheduleCron: string | null;
}

/**
 * A new listing's starting values from the workflow itself: the name,
 * description and schedule of the version being published, and the image of
 * its Library entry, if it has one.
 */
export function toAgentFormSource(
  graph: GraphModel,
  libraryAgent: LibraryAgent | null,
): AgentFormSource {
  return {
    name: graph.name,
    description: graph.description,
    imageSrc: libraryAgent?.image_url || "",
    recommendedScheduleCron: graph.recommended_schedule_cron || null,
  };
}

/**
 * The info step's starting values: the live listing's when the workflow
 * already has one (an update), otherwise the workflow's own (a new listing).
 * An update submits what the form starts with, so it takes nothing from the
 * workflow: the workflow's image or schedule would replace the listing's
 * though the creator never touched them.
 * Both ways into the dialog build them here, so they can't drift apart.
 */
export function buildInitialFormData(
  agentId: string,
  agent: AgentFormSource,
  submission: LiveStoreSubmission | undefined,
): PublishAgentInfoInitialData {
  if (!submission) {
    return {
      ...emptyModalState,
      agent_id: agentId,
      title: agent.name,
      description: agent.description,
      thumbnailSrc: agent.imageSrc,
      slug: agent.name.replace(/ /g, "-"),
      recommendedScheduleCron: agent.recommendedScheduleCron || "",
    };
  }
  const images = [...new Set(submission.image_urls || [])].filter(Boolean);
  return {
    agent_id: agentId,
    title: submission.name,
    subheader: submission.sub_heading || "",
    description: submission.description,
    instructions: submission.instructions || "",
    youtubeLink: submission.video_url || "",
    agentOutputDemo: submission.agent_output_demo_url || "",
    additionalImages: images,
    category: submission.categories?.[0] || "",
    thumbnailSrc: images[0] ?? "",
    slug: submission.slug,
    recommendedScheduleCron: submission.recommended_schedule_cron || "",
    changesSummary: submission.changes_summary || "",
  };
}

export interface SubmissionMetaItem {
  label: string;
  value: string;
  title?: string;
}

export function formatSubmissionDate(
  value: string | Date | null | undefined,
): string | null {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

const DAY_MS = 86_400_000;

export function formatRelativeDate(
  value: string | Date | null | undefined,
): string | null {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return null;

  const days = Math.floor((Date.now() - date.getTime()) / DAY_MS);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  if (days < 30) {
    const weeks = Math.floor(days / 7);
    return `${weeks} week${weeks > 1 ? "s" : ""} ago`;
  }
  if (days < 365) {
    const months = Math.floor(days / 30);
    return `${months} month${months > 1 ? "s" : ""} ago`;
  }
  const years = Math.floor(days / 365);
  return `${years} year${years > 1 ? "s" : ""} ago`;
}

function dateItem(
  label: string,
  value: string | Date | null | undefined,
): SubmissionMetaItem | null {
  const relative = formatRelativeDate(value);
  if (!relative) return null;
  return {
    label,
    value: relative,
    title: formatSubmissionDate(value) ?? undefined,
  };
}

interface SubmissionMetaArgs {
  status: SubmissionStatus | undefined;
  version: number | undefined;
  category: string | null | undefined;
  submittedAt: string | Date | null | undefined;
  reviewedAt: string | Date | null | undefined;
  runCount: number | undefined;
}

export function getSubmissionMeta({
  status,
  version,
  category,
  submittedAt,
  reviewedAt,
  runCount,
}: SubmissionMetaArgs): SubmissionMetaItem[] {
  const items: SubmissionMetaItem[] = [];

  if (typeof version === "number") {
    items.push({ label: "Version", value: `v${version}` });
  }
  if (category) {
    items.push({ label: "Category", value: category });
  }

  if (status === SubmissionStatus.APPROVED) {
    const liveSince = dateItem("Live since", reviewedAt);
    if (liveSince) items.push(liveSince);
    if (typeof runCount === "number") {
      items.push({ label: "Runs", value: runCount.toLocaleString() });
    }
  } else if (status === SubmissionStatus.REJECTED) {
    const reviewed = dateItem("Reviewed", reviewedAt);
    if (reviewed) items.push(reviewed);
  } else if (status === SubmissionStatus.DRAFT) {
    items.push({ label: "Status", value: "Not submitted yet" });
  } else {
    const submitted = dateItem("Submitted", submittedAt);
    if (submitted) items.push(submitted);
  }

  return items;
}
