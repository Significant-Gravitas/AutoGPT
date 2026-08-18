import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { RaiseAttachment } from "@/app/api/__generated__/models/raiseAttachment";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import type { RaiseAttachmentDraft } from "../../helpers";

export const MAX_ATTACHMENTS = 20;
export const DEFAULT_BUDGET_CREDITS = 500;
export const SEARCH_DEBOUNCE_MS = 300;
export const MAX_BUDGET_CREDITS = 1_000_000;
export const MAX_SEARCH_RESULTS = 3;

export const BUDGET_PRESETS = [
  { credits: 500, label: "500 credits" },
  { credits: 1000, label: "1,000 credits" },
  { credits: 0, label: "No weekly limit" },
] as const;

export interface SearchHit {
  key: string;
  name: string;
  subtitle: string;
  kind: "workflow" | "skill";
  source: "marketplace" | "library";
  id: string;
  creator?: string;
  slug?: string;
  description?: string;
}

export function scoreSearchHit(hit: SearchHit, query: string) {
  const needle = query.trim().toLowerCase();
  if (!needle) return 0;
  const name = hit.name.toLowerCase();
  if (name === needle) return 100;
  if (name.startsWith(needle)) return 80;
  if (name.includes(needle)) return 60;
  const description = hit.description?.toLowerCase() ?? "";
  if (description.includes(needle)) return 40;
  if (hit.subtitle.toLowerCase().includes(needle)) return 20;
  return 1;
}

export function limitSearchHits(hits: SearchHit[], query: string) {
  const ranked = query.trim()
    ? [...hits].sort(
        (left, right) =>
          scoreSearchHit(right, query) - scoreSearchHit(left, query),
      )
    : hits;
  return ranked.slice(0, MAX_SEARCH_RESULTS);
}

export function parseCredits(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  if (!/^\d+$/.test(trimmed)) return null;
  const credits = Number(trimmed);
  if (!Number.isInteger(credits)) return null;
  if (credits < 0 || credits > MAX_BUDGET_CREDITS) return null;
  return credits;
}

export function marketplaceKey(creator: string, slug: string) {
  return `${creator.toLowerCase()}/${slug}`;
}

export type KitSearchScope = "marketplace" | "skills";

export function combineSearchHits({
  query,
  storeAgents,
  libraryAgents,
  skills,
  scope,
}: {
  query: string;
  storeAgents: StoreAgent[];
  libraryAgents: LibraryAgent[];
  skills: CopilotSkillInfo[];
  scope: KitSearchScope;
}): SearchHit[] {
  const hits: SearchHit[] = [];
  const kind = scope === "marketplace" ? "workflow" : "skill";
  for (const agent of storeAgents) {
    hits.push(marketplaceHit(agent, kind));
  }
  if (scope === "marketplace") {
    for (const agent of libraryAgents) {
      hits.push({
        key: `library:workflow:${agent.id}`,
        name: agent.name,
        subtitle: "Library workflow",
        kind: "workflow",
        source: "library",
        id: agent.id,
      });
    }
    return limitSearchHits(hits, query);
  }
  const needle = query.trim().toLowerCase();
  for (const skill of skills) {
    if (!skillMatches(skill, needle)) continue;
    hits.push({
      key: `library:skill:${skill.name.toLowerCase()}`,
      name: skill.name,
      subtitle: "Library skill",
      kind: "skill",
      source: "library",
      id: skill.name,
      description: skill.description,
    });
  }
  return limitSearchHits(hits, query);
}

export function skillMatches(skill: CopilotSkillInfo, needle: string) {
  if (!needle) return true;
  return (
    skill.name.toLowerCase().includes(needle) ||
    skill.description.toLowerCase().includes(needle)
  );
}

export function isHitSelected(
  attachments: RaiseAttachmentDraft[],
  hit: SearchHit,
) {
  return attachments.some((attachment) => hitsMatch(attachment, hit));
}

export function hitsMatch(attachment: RaiseAttachmentDraft, hit: SearchHit) {
  if (attachment.kind !== hit.kind || attachment.source !== hit.source) {
    return false;
  }
  if (hit.source === "marketplace") {
    const listing =
      hit.creator && hit.slug ? marketplaceKey(hit.creator, hit.slug) : null;
    return (
      attachment.id === hit.id ||
      (Boolean(listing) && attachment.marketplaceKey === listing)
    );
  }
  if (hit.kind === "skill") {
    return attachment.id.toLowerCase() === hit.id.toLowerCase();
  }
  return attachment.id === hit.id;
}

export function toRaiseAttachments(
  drafts: RaiseAttachmentDraft[],
): RaiseAttachment[] {
  return drafts.map((draft) => ({
    kind: draft.kind,
    source: draft.source,
    id: draft.id,
  }));
}

export function failedAttachmentMessage(
  failures: { kind: string; source: string; id: string; reason: string }[],
  drafts: RaiseAttachmentDraft[],
) {
  return failures
    .map((failure) => {
      const name =
        drafts.find(
          (draft) =>
            draft.kind === failure.kind &&
            draft.source === failure.source &&
            draft.id === failure.id,
        )?.name ?? failure.id;
      const reason =
        failure.reason === "unavailable"
          ? "is no longer available"
          : "couldn't be installed";
      return `${name} ${reason}`;
    })
    .join(". ");
}

function marketplaceHit(
  agent: StoreAgent,
  kind: "workflow" | "skill",
): SearchHit {
  return {
    key: `marketplace:${kind}:${agent.creator.toLowerCase()}/${agent.slug}`,
    name: agent.agent_name,
    subtitle:
      kind === "workflow" ? "Marketplace workflow" : "Marketplace skill",
    kind,
    source: "marketplace",
    id: "",
    creator: agent.creator,
    slug: agent.slug,
  };
}
