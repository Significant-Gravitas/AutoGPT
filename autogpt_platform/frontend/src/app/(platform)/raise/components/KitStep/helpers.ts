import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import type { RaiseAttachment } from "@/app/api/__generated__/models/raiseAttachment";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { parseUsdToCredits } from "@/lib/credits";
import type { RaiseAttachmentDraft } from "../../helpers";

export const MAX_ATTACHMENTS = 20;
export const DEFAULT_BUDGET_CREDITS = 500;
export const SEARCH_DEBOUNCE_MS = 300;
export const MAX_BUDGET_CREDITS = 1_000_000;
export const MAX_SEARCH_RESULTS = 3;

export const BUDGET_PRESETS = [
  { credits: 500, label: "$5 / week" },
  { credits: 1000, label: "$10 / week" },
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

export function parseBudget(value: string): number | null {
  return parseUsdToCredits(value, MAX_BUDGET_CREDITS);
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
  marketplaceSkills,
  scope,
}: {
  query: string;
  storeAgents: StoreAgent[];
  libraryAgents: LibraryAgent[];
  skills: CopilotSkillInfo[];
  marketplaceSkills: MarketplaceSkill[];
  scope: KitSearchScope;
}): SearchHit[] {
  const hits: SearchHit[] = [];
  if (scope === "marketplace") {
    for (const agent of storeAgents) {
      hits.push(marketplaceWorkflowHit(agent));
    }
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
  const libraryHits = skills
    .filter((skill) => skillMatches(skill, needle))
    .map(
      (skill): SearchHit => ({
        key: `library:skill:${skill.name.toLowerCase()}`,
        name: skill.name,
        subtitle: "Library skill",
        kind: "skill",
        source: "library",
        id: skill.name,
        description: skill.description,
      }),
    );
  // Browsing with nothing typed, the Hub asks for exactly as many listings as
  // there are slots, so pushing it first would bury the user's own skills —
  // the half they are likeliest to be after. Interleaving keeps both visible.
  // A typed query is ranked instead, so order in equals order out there.
  return limitSearchHits(
    interleave(libraryHits, marketplaceSkills.map(marketplaceSkillHit)),
    query,
  );
}

function interleave(left: SearchHit[], right: SearchHit[]): SearchHit[] {
  const merged: SearchHit[] = [];
  for (let i = 0; i < Math.max(left.length, right.length); i++) {
    if (i < left.length) merged.push(left[i]);
    if (i < right.length) merged.push(right[i]);
  }
  return merged;
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

function marketplaceWorkflowHit(agent: StoreAgent): SearchHit {
  return {
    key: `marketplace:workflow:${agent.creator.toLowerCase()}/${agent.slug}`,
    name: agent.agent_name,
    subtitle: "Marketplace workflow",
    kind: "workflow",
    source: "marketplace",
    // Resolved to a store listing version id when the row is added.
    id: "",
    creator: agent.creator,
    slug: agent.slug,
  };
}

// A Hub skill listing is addressed by its slug, so nothing needs resolving.
function marketplaceSkillHit(skill: MarketplaceSkill): SearchHit {
  return {
    key: `marketplace:skill:${skill.slug}`,
    name: skill.title,
    subtitle: "Marketplace skill",
    kind: "skill",
    source: "marketplace",
    id: skill.slug,
    description: skill.description,
  };
}
