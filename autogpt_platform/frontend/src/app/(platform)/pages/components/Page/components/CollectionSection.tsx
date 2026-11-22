import { CollectionBoard } from "@/app/(platform)/collections/components/CollectionBoard/CollectionBoard";
import { CollectionCards } from "@/app/(platform)/collections/components/CollectionCards/CollectionCards";
import { CollectionTable } from "@/app/(platform)/collections/components/CollectionTable/CollectionTable";
import { ChartBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/ChartBlockView";
import { StatsBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/StatsBlockView";
import type { PageCollectionBoard } from "@/app/api/__generated__/models/pageCollectionBoard";
import type { PageCollectionCards } from "@/app/api/__generated__/models/pageCollectionCards";
import type { PageCollectionChart } from "@/app/api/__generated__/models/pageCollectionChart";
import type { PageCollectionNumbers } from "@/app/api/__generated__/models/pageCollectionNumbers";
import type { PageCollectionTable } from "@/app/api/__generated__/models/pageCollectionTable";
import { Text } from "@/components/atoms/Text/Text";
import type { PageLive } from "../Page";

interface Props {
  section:
    | PageCollectionTable
    | PageCollectionNumbers
    | PageCollectionBoard
    | PageCollectionCards
    | PageCollectionChart;
  /** Absent in a chat preview, where nothing is written or run. */
  live?: PageLive;
}

export function CollectionSection({ section, live }: Props) {
  const onRunStarted = live?.onRunStarted;
  switch (section.kind) {
    case "collection_numbers":
      return <StatsBlockView block={section.block} />;
    case "collection_chart":
      return section.block ? (
        <ChartBlockView block={section.block} />
      ) : (
        <Text variant="body" tone="secondary">
          Nothing to chart yet.
        </Text>
      );
    case "collection_board":
      return (
        <CollectionBoard
          section={section}
          canEdit={Boolean(live)}
          onRunStarted={onRunStarted}
        />
      );
    case "collection_cards":
      return (
        <CollectionCards
          section={section}
          canRun={Boolean(live)}
          onRunStarted={onRunStarted}
        />
      );
    case "collection_table":
      return (
        <div className="flex flex-col gap-2">
          <CollectionTable
            collectionID={section.collection_id}
            columns={section.columns}
            rows={section.rows}
            total={section.total}
            filter={section.filter ?? []}
            sort={section.sort ?? null}
            readOnly={section.read_only || !live}
            buttons={section.buttons ?? []}
            onRunStarted={onRunStarted}
          />
          {section.dropped_columns?.length ? (
            <Text variant="small" tone="secondary">
              No longer in the collection: {section.dropped_columns.join(", ")}
            </Text>
          ) : null}
        </div>
      );
  }
}
