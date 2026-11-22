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

const IDEAS = { collection_id: "ideas", collection_name: "Content ideas" };
const KEYWORD = { field: "keyword", type: "text" } as const;
const VOLUME = { field: "volume", type: "number" } as const;
const DRAFT_BUTTON = {
  label: "Draft",
  library_agent_id: "draft",
  graph_id: "draft-graph",
  graph_version: 1,
  inputs: { keyword: "keyword" },
};

function idea(id: string, keyword: string, volume: number, status: string) {
  return {
    id,
    values: { keyword, volume, status },
    pending: {},
    written_by: "workflow:keyword-ideas",
    revision: 1,
    created_at: "2026-11-20T09:00:00Z",
    updated_at: "2026-11-21T09:00:00Z",
  };
}

const DESK = {
  id: "desk",
  title: "Content desk",
  revision: 1,
  has_previous: false,
  sections: [
    {
      kind: "collection_board",
      index: 0,
      ...IDEAS,
      group_by: "status",
      card_title: KEYWORD,
      card_fields: [VOLUME],
      read_only: false,
      buttons: [DRAFT_BUTTON],
      columns: [
        {
          option: "New",
          total: 2,
          rows: [
            idea("i1", "cohort retention chart", 1900, "New"),
            idea("i2", "weekly cohort table", 480, "New"),
          ],
        },
        {
          option: "Drafted",
          total: 1,
          rows: [idea("i3", "cohort analysis example", 1300, "Drafted")],
        },
        {
          option: "Posted",
          total: 1,
          rows: [idea("i4", "cohort churn formula", 390, "Posted")],
        },
      ],
    },
    {
      kind: "collection_cards",
      index: 1,
      ...IDEAS,
      title: "Biggest ideas",
      card_title: KEYWORD,
      card_fields: [VOLUME, { field: "status", type: "select" }],
      rows: [
        idea("i1", "cohort retention chart", 1900, "New"),
        idea("i3", "cohort analysis example", 1300, "Drafted"),
      ],
      total: 4,
      buttons: [],
    },
    {
      kind: "collection_chart",
      index: 2,
      ...IDEAS,
      title: "Search volume by status",
      block: {
        type: "chart",
        kind: "bar",
        x_labels: ["New", "Drafted", "Posted"],
        series: [{ name: "volume", values: [2380, 1300, 390] }],
      },
    },
  ],
} as unknown as ResolvedPage;

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
      <Page page={DESK} />
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
