import type { PageButton } from "@/app/api/__generated__/models/pageButton";
import type { PageTableColumn } from "@/app/api/__generated__/models/pageTableColumn";
import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { cellText, type TableRow } from "../CollectionTable/helpers";
import { RowButtons } from "../RowButtons/RowButtons";
import type { useRowButtons } from "../RowButtons/useRowButtons";

interface Props {
  row: TableRow;
  title: PageTableColumn;
  fields: PageTableColumn[];
  buttons: PageButton[];
  actions: ReturnType<typeof useRowButtons>;
  canRun: boolean;
  footer?: ReactNode;
}

export function RowCard({
  row,
  title,
  fields,
  buttons,
  actions,
  canRun,
  footer,
}: Props) {
  const heading = cellText(title, row.values[title.field]) || "Untitled";
  return (
    <div className="flex flex-col gap-2 rounded-xl bg-surface p-3 smooth-shadow-ring-sm">
      <Text variant="body-medium" className="break-words">
        {heading}
      </Text>
      {fields.length ? (
        <dl className="flex flex-col gap-0.5">
          {fields.map((field) => (
            <div key={field.field} className="flex justify-between gap-3">
              <dt>
                <Text as="span" variant="small" tone="secondary">
                  {field.field}
                </Text>
              </dt>
              <dd>
                <Text
                  as="span"
                  variant="small"
                  tabular={field.type === "number"}
                >
                  {cellText(field, row.values[field.field]) || "–"}
                </Text>
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
      {canRun ? (
        <RowButtons
          buttons={buttons}
          row={row}
          rowLabel={heading}
          actions={actions}
        />
      ) : null}
      {footer}
    </div>
  );
}
