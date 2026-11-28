import { CollectionTable } from "@/app/(platform)/collections/components/CollectionTable/CollectionTable";
import { StatsBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/StatsBlockView";
import type { PageCollectionNumbers } from "@/app/api/__generated__/models/pageCollectionNumbers";
import type { PageCollectionTable } from "@/app/api/__generated__/models/pageCollectionTable";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  section: PageCollectionTable | PageCollectionNumbers;
  /** Off in a chat preview, where nothing is written. */
  canEdit: boolean;
}

export function CollectionSection({ section, canEdit }: Props) {
  if (section.kind === "collection_numbers") {
    return <StatsBlockView block={section.block} />;
  }
  return (
    <div className="flex flex-col gap-2">
      <CollectionTable
        collectionID={section.collection_id}
        columns={section.columns}
        rows={section.rows}
        total={section.total}
        filter={section.filter ?? []}
        sort={section.sort ?? null}
        readOnly={section.read_only || !canEdit}
      />
      {section.dropped_columns?.length ? (
        <Text variant="small" tone="secondary">
          No longer in the collection: {section.dropped_columns.join(", ")}
        </Text>
      ) : null}
    </div>
  );
}
