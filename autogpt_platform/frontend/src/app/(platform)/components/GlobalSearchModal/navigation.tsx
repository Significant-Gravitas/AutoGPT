import type { SearchCommandBucket } from "@/components/organisms/SearchCommandModal/helpers";
import {
  ChatCircleIcon,
  CreditCardIcon,
  GearIcon,
  HammerIcon,
  StackIcon,
  StorefrontIcon,
  UserIcon,
} from "@phosphor-icons/react";
import type { ComponentType } from "react";

export const NAV_BUCKET_KEY = "navigation";
export const NAV_BUCKET_LABEL = "Navigate";

// Idle (empty query) only previews a few destinations; typing reveals
// the full filtered set.
const IDLE_NAV_LIMIT = 4;

interface NavTarget {
  id: string;
  title: string;
  href: string;
  // Extra terms that should match this target beyond its title (e.g.
  // "agents" routing to the library page).
  keywords: string[];
  icon: ComponentType<{ className?: string }>;
}

// Namespaced ids so they never collide with API result ids in the
// shared ``itemsById`` map / keyboard-nav flattening.
const NAV_TARGETS: NavTarget[] = [
  {
    id: "nav:builder",
    title: "Builder",
    href: "/build",
    keywords: ["build", "editor", "graph", "workflow", "create"],
    icon: HammerIcon,
  },
  {
    id: "nav:library",
    title: "Library",
    href: "/library",
    keywords: ["agents", "my agents"],
    icon: StackIcon,
  },
  {
    id: "nav:marketplace",
    title: "Marketplace",
    href: "/marketplace",
    keywords: ["store", "explore", "templates"],
    icon: StorefrontIcon,
  },
  {
    id: "nav:chat",
    title: "Chat",
    href: "/copilot",
    keywords: ["copilot", "home", "assistant"],
    icon: ChatCircleIcon,
  },
  {
    id: "nav:settings",
    title: "Settings",
    href: "/settings",
    keywords: ["preferences", "account"],
    icon: GearIcon,
  },
  {
    id: "nav:billing",
    title: "Billing",
    href: "/settings/billing",
    keywords: ["credits", "payment", "wallet", "subscription"],
    icon: CreditCardIcon,
  },
  {
    id: "nav:profile",
    title: "Profile",
    href: "/settings/profile",
    keywords: ["account", "me"],
    icon: UserIcon,
  },
];

const HREF_BY_ID = new Map(
  NAV_TARGETS.map((target) => [target.id, target.href]),
);

export function getNavigationHref(id: string): string | undefined {
  return HREF_BY_ID.get(id);
}

function isExactTitleMatch(target: NavTarget, query: string): boolean {
  return target.title.toLocaleLowerCase() === query;
}

// True when ``currentPath`` is the target itself or any page nested under
// it (e.g. ``/settings/billing`` is "on" ``/settings``). The trailing
// slash keeps unrelated routes that merely share a prefix (``/settings-x``)
// from matching.
function isCurrentDestination(
  target: NavTarget,
  currentPath: string | null,
): boolean {
  if (!currentPath) return false;
  return (
    currentPath === target.href || currentPath.startsWith(`${target.href}/`)
  );
}

function matchesQuery(target: NavTarget, query: string): boolean {
  if (target.title.toLocaleLowerCase().includes(query)) return true;
  return target.keywords.some((keyword) =>
    keyword.toLocaleLowerCase().includes(query),
  );
}

/**
 * Build the static "Navigate" bucket filtered by ``query``. The
 * destination matching ``currentPath`` is dropped — there's no point
 * navigating to the page you're already on. An empty query previews a
 * few destinations alongside the idle recent-items list; typing reveals
 * the full filtered set. ``isExactMatch`` lets the caller hoist
 * navigation above the search results when the user typed a destination
 * name verbatim (e.g. "Builder").
 */
export function buildNavigationBucket(
  query: string,
  currentPath: string | null,
): {
  bucket: SearchCommandBucket | null;
  isExactMatch: boolean;
} {
  const normalized = query.trim().toLocaleLowerCase();

  const available = NAV_TARGETS.filter(
    (target) => !isCurrentDestination(target, currentPath),
  );
  const matches = normalized
    ? available.filter((target) => matchesQuery(target, normalized))
    : available.slice(0, IDLE_NAV_LIMIT);
  if (matches.length === 0) return { bucket: null, isExactMatch: false };

  const isExactMatch = normalized
    ? matches.some((target) => isExactTitleMatch(target, normalized))
    : false;

  const bucket: SearchCommandBucket = {
    key: NAV_BUCKET_KEY,
    label: NAV_BUCKET_LABEL,
    items: matches.map((target) => ({
      id: target.id,
      title: target.title,
      icon: target.icon,
    })),
  };

  return { bucket, isExactMatch };
}
