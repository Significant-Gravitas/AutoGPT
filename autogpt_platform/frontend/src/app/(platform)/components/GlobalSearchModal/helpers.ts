import type { GlobalSearchResponse } from "@/app/api/__generated__/models/globalSearchResponse";
import type { SearchResultItem } from "@/app/api/__generated__/models/searchResultItem";
import type { SearchResultItemType } from "@/app/api/__generated__/models/searchResultItemType";
import type {
  SearchCommandBucket,
  SearchCommandItem,
} from "@/components/organisms/SearchCommandModal/helpers";
import {
  BookOpenIcon,
  ChatCircleIcon,
  FileIcon,
  StorefrontIcon,
} from "@phosphor-icons/react";
import type { ComponentType } from "react";

export type BucketKey = "agents" | "files" | "chats";

const ICON_BY_TYPE: Record<
  SearchResultItemType,
  ComponentType<{ className?: string }>
> = {
  library_agent: BookOpenIcon,
  store_agent: StorefrontIcon,
  workspace_file: FileIcon,
  chat_session: ChatCircleIcon,
};

const BUCKET_LABEL: Record<BucketKey, string> = {
  agents: "Agents",
  files: "Files",
  chats: "Chats",
};

const BUCKET_ORDER: BucketKey[] = ["chats", "agents", "files"];

function toCommandItem(item: SearchResultItem): SearchCommandItem {
  return {
    id: item.id,
    title: item.title,
    subtitle: item.subtitle ?? null,
    icon: ICON_BY_TYPE[item.type],
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
    for (const apiItem of apiItems) itemsById.set(apiItem.id, apiItem);
    return {
      key,
      label: BUCKET_LABEL[key],
      items: apiItems.map(toCommandItem),
    };
  });

  return { buckets, itemsById };
}
