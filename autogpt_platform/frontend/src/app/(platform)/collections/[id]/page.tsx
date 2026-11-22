"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Flag } from "@/services/feature-flags/use-get-flag";
import { withFeatureFlag } from "@/services/feature-flags/with-feature-flag";
import NextLink from "next/link";
import { useParams } from "next/navigation";
import { CollectionTable } from "../components/CollectionTable/CollectionTable";
import { COLLECTIONS_LIST_HREF, useCollectionView } from "./useCollectionView";

function CollectionView() {
  const { id } = useParams<{ id: string }>();
  const view = useCollectionView(id);
  const collection = view.collection;

  if (view.isMissing) {
    return (
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-3 p-4 sm:p-6">
        <Text variant="h3" as="h1">
          This collection is no longer available
        </Text>
        <NextLink href={COLLECTIONS_LIST_HREF}>
          <Text variant="body" tone="secondary">
            Back to your collections
          </Text>
        </NextLink>
      </main>
    );
  }
  if (view.isLoading || !collection) {
    return (
      <main className="mx-auto w-full max-w-5xl p-4 sm:p-6">
        <Skeleton className="h-64 w-full rounded-2xl" />
      </main>
    );
  }

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-col gap-5 p-4 sm:p-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-col gap-1">
          <Text variant="h2" as="h1" className="break-words">
            {collection.name}
          </Text>
          <Text variant="small" tone="secondary">
            {collection.row_count?.toLocaleString("en-US")} rows ·{" "}
            {collection.fields.length} fields
            {collection.key_field
              ? ` · each row is named by ${collection.key_field}`
              : ""}
          </Text>
          {view.limits ? (
            <Text variant="small" tone="secondary">
              Holds up to {view.limits.rows.toLocaleString("en-US")} rows and{" "}
              {view.limits.fields} fields, {view.limits.row_bytes / 1024} KB a
              row.
            </Text>
          ) : null}
        </div>
        <Button
          size="small"
          variant="ghost-danger"
          onClick={() => view.setConfirmingDelete(true)}
        >
          Delete collection
        </Button>
      </header>
      <CollectionTable
        collectionID={collection.id}
        columns={collection.fields.map((field) => ({
          field: field.name,
          type: field.type,
          options: field.options,
          unit: field.unit,
        }))}
        rows={view.rows}
        total={view.total}
        readOnly={false}
      />
      <Dialog
        controlled={{
          isOpen: view.confirmingDelete,
          set: view.setConfirmingDelete,
        }}
        styling={{ maxWidth: "28rem" }}
        title={`Delete ${collection.name}?`}
      >
        <Dialog.Content>
          <div className="flex flex-col gap-3">
            <Text variant="body">
              Its {collection.row_count} rows go with it, and workflows that
              write to it will start a new, empty one.
            </Text>
            <Button
              variant="destructive"
              loading={view.isDeleting}
              onClick={view.deleteCollection}
            >
              Delete
            </Button>
          </div>
        </Dialog.Content>
      </Dialog>
    </main>
  );
}

export default withFeatureFlag(CollectionView, Flag.COLLECTIONS);
