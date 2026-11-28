import { CollectionTable } from "@/app/(platform)/collections/components/CollectionTable/CollectionTable";
import { Page } from "@/app/(platform)/pages/components/Page/Page";
import type { CollectionRowRecord } from "@/app/api/__generated__/models/collectionRowRecord";
import type { PageTableColumn } from "@/app/api/__generated__/models/pageTableColumn";
import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";

const LEDGER = { collection_id: "invoices", collection_name: "Invoices" };
const COLUMNS: PageTableColumn[] = [
  { field: "invoice_no", type: "text" },
  { field: "supplier", type: "text" },
  { field: "amount", type: "number", unit: "$" },
  {
    field: "status",
    type: "select",
    options: ["Booked", "Needs review", "Rejected"],
  },
  { field: "needs_review", type: "checkbox" },
];

function row(
  id: string,
  values: Record<string, unknown>,
  pending: Record<string, unknown> = {},
): CollectionRowRecord {
  return {
    id,
    key: String(values.invoice_no),
    values,
    pending,
    manual_fields: Object.keys(pending),
    written_by: "workflow:invoice-intake",
    revision: 2,
    created_at: new Date("2026-11-20T09:00:00Z"),
    updated_at: new Date("2026-11-21T09:00:00Z"),
  };
}

const ROWS = [
  row(
    "r3",
    {
      invoice_no: "INV-2031",
      supplier: "Lumen Forklift Lease",
      amount: 340,
      status: "Needs review",
      needs_review: true,
    },
    { amount: 3500 },
  ),
  row("r2", {
    invoice_no: "INV-2030",
    supplier: "Nordkai Customs",
    amount: 1180.5,
    status: "Booked",
    needs_review: false,
  }),
  row("r1", {
    invoice_no: "INV-2029",
    supplier: "Harbour Pallets BV",
    amount: 640,
    status: "Booked",
    needs_review: false,
  }),
];

const SAMPLE = {
  id: "ledger",
  title: "Invoices",
  revision: 1,
  has_previous: false,
  sections: [
    {
      kind: "collection_numbers",
      index: 0,
      ...LEDGER,
      block: {
        type: "stats",
        items: [
          { label: "Booked this month", value: "$1,820.50" },
          { label: "Needs review", value: "1" },
        ],
      },
    },
    {
      kind: "collection_table",
      index: 1,
      ...LEDGER,
      title: "This month",
      columns: COLUMNS,
      dropped_columns: ["approved_by"],
      rows: ROWS,
      total: 3,
      filter: [],
      sort: { field: "amount", direction: "desc" },
      read_only: true,
      key_field: "invoice_no",
    },
  ],
} as unknown as ResolvedPage;

export function CollectionSamples() {
  return (
    <div className="flex max-w-4xl flex-col gap-8">
      <Page page={SAMPLE} />
      <CollectionTable
        collectionID="invoices"
        columns={COLUMNS}
        rows={ROWS}
        total={140}
        readOnly={false}
      />
    </div>
  );
}
