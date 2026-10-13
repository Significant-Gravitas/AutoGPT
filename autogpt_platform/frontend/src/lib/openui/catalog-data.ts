import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import {
  columnsSchema,
  identifier,
  rowsSchema,
  shortText,
  validateRows,
} from "./data-values";

export const DataGrid = defineComponent({
  name: "DataGrid",
  description:
    "Typed, searchable, sortable, paginated data inside Form. Up to 100 stable-id rows and eight columns. A numeric column may have calculation:{operation:add/subtract/multiply/divide,left:columnKey,right:columnKey}; both inputs must be non-calculated numeric columns. Calculated cells are read-only and always recompute (e.g. quantity times unit price); omit their values in rows. Missing inputs/zero divisor produce unknown, never stale totals. selection is none/single/multiple; selected contains row ids. Editable cells validate locally. allowRows adds/removes rows when editable. CSV export uses current rows, not only the visible page. Publishes name_rows with typed cells and name_selected with selected ids. Do not use for approvals or executed-action status. Dashboard can bind to this grid by form and field.",
  props: z
    .object({
      name: identifier,
      title: shortText,
      columns: columnsSchema,
      rows: rowsSchema,
      editable: z.boolean(),
      selection: z.enum(["none", "single", "multiple"]),
      selected: z.array(identifier).max(100),
      allowRows: z.boolean().optional(),
    })
    .superRefine((props, context) => {
      for (const message of validateRows(props.columns, props.rows)) {
        if (!props.editable || message.startsWith("Unknown column"))
          context.addIssue({ code: "custom", message });
      }
      if (
        new Set(props.selected).size !== props.selected.length ||
        props.selected.some((id) => !props.rows.some((row) => row.id === id)) ||
        (props.selection === "single" && props.selected.length > 1) ||
        (props.selection === "none" && props.selected.length)
      )
        context.addIssue({
          code: "custom",
          message:
            "Selected ids must be unique existing rows and match the selection mode.",
        });
    }),
  component: null,
});
