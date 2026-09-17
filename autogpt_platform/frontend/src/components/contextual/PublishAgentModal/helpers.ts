import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";

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
