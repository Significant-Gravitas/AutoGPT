import type { PageButton } from "@/app/api/__generated__/models/pageButton";
import type { TableRow } from "../CollectionTable/helpers";

/** The workflow's inputs, each taken from the row field the button names. */
export function runInputs(button: PageButton, row: TableRow) {
  return Object.fromEntries(
    Object.entries(button.inputs).map(([input, field]) => [
      input,
      row.values[field] ?? null,
    ]),
  );
}
