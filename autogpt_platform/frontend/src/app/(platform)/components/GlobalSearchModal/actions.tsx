import type { SearchCommandBucket } from "@/components/organisms/SearchCommandModal/helpers";
import { CopyIcon } from "@phosphor-icons/react";
import type { ComponentType } from "react";

export const ACTIONS_BUCKET_KEY = "actions";
export const ACTIONS_BUCKET_LABEL = "Actions";

export const COPY_USER_ID_ACTION = "action:copy-user-id";

interface ActionTarget {
  id: string;
  title: string;
  // Extra terms that match this action beyond its title.
  keywords: string[];
  icon: ComponentType<{ className?: string }>;
}

// Namespaced ids so they never collide with API result / navigation ids
// in the shared keyboard-nav flattening.
const ACTION_TARGETS: ActionTarget[] = [
  {
    id: COPY_USER_ID_ACTION,
    title: "Copy user ID",
    keywords: ["copy", "user", "id", "uid", "account"],
    icon: CopyIcon,
  },
];

function matchesQuery(target: ActionTarget, query: string): boolean {
  if (target.title.toLocaleLowerCase().includes(query)) return true;
  return target.keywords.some((keyword) =>
    keyword.toLocaleLowerCase().includes(query),
  );
}

function isExactTitleMatch(target: ActionTarget, query: string): boolean {
  return target.title.toLocaleLowerCase() === query;
}

/**
 * Build the static "Actions" bucket filtered by ``query``. Actions run a
 * side effect (e.g. copy to clipboard) rather than navigate. An empty
 * query lists every action so they stay available on the idle list.
 * ``isExactMatch`` lets the caller hoist actions above the search
 * results when the user typed an action name verbatim.
 */
export function buildActionsBucket(query: string): {
  bucket: SearchCommandBucket | null;
  isExactMatch: boolean;
} {
  const normalized = query.trim().toLocaleLowerCase();

  const matches = normalized
    ? ACTION_TARGETS.filter((target) => matchesQuery(target, normalized))
    : ACTION_TARGETS;
  if (matches.length === 0) return { bucket: null, isExactMatch: false };

  const isExactMatch = normalized
    ? matches.some((target) => isExactTitleMatch(target, normalized))
    : false;

  const bucket: SearchCommandBucket = {
    key: ACTIONS_BUCKET_KEY,
    label: ACTIONS_BUCKET_LABEL,
    items: matches.map((target) => ({
      id: target.id,
      title: target.title,
      icon: target.icon,
    })),
  };

  return { bucket, isExactMatch };
}
