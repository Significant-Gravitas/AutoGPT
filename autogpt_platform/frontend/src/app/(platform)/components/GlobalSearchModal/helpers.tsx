import type { GlobalSearchResponse } from "@/app/api/__generated__/models/globalSearchResponse";
import type { SearchResultItem } from "@/app/api/__generated__/models/searchResultItem";
import type { SearchResultItemType } from "@/app/api/__generated__/models/searchResultItemType";
import type {
  SearchCommandBucket,
  SearchCommandItem,
} from "@/components/organisms/SearchCommandModal/helpers";
import { cn } from "@/lib/utils";
import {
  ChatCircleIcon,
  FileIcon,
  StorefrontIcon,
  TreeStructureIcon,
} from "@phosphor-icons/react";
import Image from "next/image";
import type { ComponentType } from "react";
import {
  type PlatformLogo,
  resolvePlatformLogo,
} from "@/app/(platform)/copilot/components/ChatOriginIcon/platformLogos";

export type BucketKey = "agents" | "files" | "chats";

const ICON_BY_TYPE: Record<
  SearchResultItemType,
  ComponentType<{ className?: string }>
> = {
  // Workflow / node-graph cue maps better to "an agent" than a robot
  // mascot — agents in this product are visual graphs of nodes, not
  // anthropomorphic assistants.
  library_agent: TreeStructureIcon,
  store_agent: StorefrontIcon,
  workspace_file: FileIcon,
  chat_session: ChatCircleIcon,
};

// Chat sessions opened from external platforms (Discord, Slack, …)
// carry their origin in ``metadata.source_platform``. We reuse the
// ChatOriginIcon platform-logo table so the search row shows the same
// branded image as the sidebar / chat header — consistent visual
// language across surfaces.
function makePlatformIconComponent(
  logo: PlatformLogo,
): ComponentType<{ className?: string }> {
  function PlatformIcon({ className }: { className?: string }) {
    return (
      <Image
        src={logo.src}
        alt={logo.name}
        width={16}
        height={16}
        loading="lazy"
        className={cn("size-4 object-contain", className)}
      />
    );
  }
  PlatformIcon.displayName = `PlatformIcon(${logo.name})`;
  return PlatformIcon;
}

function resolveIcon(
  item: SearchResultItem,
): ComponentType<{ className?: string }> {
  if (item.type === "chat_session") {
    const logo = resolvePlatformLogo(
      item.metadata?.source_platform as string | undefined,
    );
    if (logo) return makePlatformIconComponent(logo);
  }
  return ICON_BY_TYPE[item.type];
}

const BUCKET_LABEL: Record<BucketKey, string> = {
  agents: "Agents",
  files: "Files",
  chats: "Chats",
};

const BUCKET_ORDER: BucketKey[] = ["chats", "agents", "files"];

// Namespace the key by type: raw IDs can collide across buckets (e.g. a
// library agent and a chat session sharing an ID), which would clobber
// ``itemsById`` and route a click to the wrong target.
function itemKey(item: SearchResultItem): string {
  return `${item.type}:${item.id}`;
}

function toCommandItem(item: SearchResultItem): SearchCommandItem {
  return {
    id: itemKey(item),
    title: item.title,
    subtitle: item.subtitle ?? null,
    icon: resolveIcon(item),
  };
}

/**
 * Adapt the bucketed API response into the generic ``SearchCommandModal``
 * shape. Bucket order matches ``BUCKET_ORDER``; the original
 * ``SearchResultItem`` for each row is returned alongside via
 * ``itemsById`` so the container can route on the original ``type``
 * after a click.
 */
export function buildBucketsFromResponse(
  response: GlobalSearchResponse | undefined,
): {
  buckets: SearchCommandBucket[];
  itemsById: Map<string, SearchResultItem>;
} {
  const itemsById = new Map<string, SearchResultItem>();
  if (!response) return { buckets: [], itemsById };

  const buckets: SearchCommandBucket[] = BUCKET_ORDER.map((key) => {
    const apiItems = response[key] ?? [];
    for (const apiItem of apiItems) itemsById.set(itemKey(apiItem), apiItem);
    return {
      key,
      label: BUCKET_LABEL[key],
      items: apiItems.map(toCommandItem),
    };
  });

  return { buckets, itemsById };
}
