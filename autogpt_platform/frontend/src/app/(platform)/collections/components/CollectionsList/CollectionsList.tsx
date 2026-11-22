"use client";

import { useListCollections } from "@/app/api/__generated__/endpoints/collections/collections";
import type { CollectionInfo } from "@/app/api/__generated__/models/collectionInfo";
import { okData } from "@/app/api/helpers";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { formatDate } from "@/lib/utils/time";
import NextLink from "next/link";

export function CollectionsList() {
  const query = useListCollections({ query: { select: okData } });
  const collections = query.data?.collections ?? [];

  if (query.isLoading) return <Skeleton className="h-40 w-full rounded-2xl" />;
  if (!collections.length) {
    return (
      <div className="flex flex-col items-start gap-3 rounded-2xl bg-surface p-6 smooth-shadow-ring-sm">
        <Text variant="body-medium">No collections yet</Text>
        <Text variant="body" tone="secondary">
          A collection is a table of your own. Your workflows fill one with the
          collection blocks, and Otto keeps one when you ask it to track
          something.
        </Text>
      </div>
    );
  }
  return (
    <ul className="flex flex-col gap-2">
      {collections.map((collection) => (
        <li key={collection.id}>
          <CollectionRow collection={collection} />
        </li>
      ))}
    </ul>
  );
}

function CollectionRow({ collection }: { collection: CollectionInfo }) {
  const rows = collection.row_count ?? 0;
  return (
    <NextLink
      href={`/collections/${collection.id}`}
      className="flex items-center justify-between gap-3 rounded-2xl bg-surface p-4 smooth-shadow-ring-sm hover:bg-surface-muted"
    >
      <span className="flex min-w-0 flex-col gap-0.5">
        <Text variant="body-medium" className="truncate">
          {collection.name}
        </Text>
        <Text variant="small" tone="secondary">
          {rows.toLocaleString("en-US")} {rows === 1 ? "row" : "rows"} ·{" "}
          {collection.fields.length} fields
        </Text>
      </span>
      <Text variant="small" tone="secondary" className="shrink-0">
        {formatDate(new Date(collection.updated_at))}
      </Text>
    </NextLink>
  );
}
