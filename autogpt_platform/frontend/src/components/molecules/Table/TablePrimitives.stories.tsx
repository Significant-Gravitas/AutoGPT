import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { Badge } from "@/components/atoms/Badge/Badge";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableFooter,
  TableHead,
  TableHeader,
  TableRow,
} from "./TablePrimitives";

const ROWS = [
  { id: "INV-001", status: "success", method: "Card", amount: "$250.00" },
  { id: "INV-002", status: "warning", method: "PayPal", amount: "$150.00" },
  { id: "INV-003", status: "error", method: "Transfer", amount: "$350.00" },
] as const;

const meta = {
  title: "Molecules/Table/Primitives",
  component: Table,
  tags: ["autodocs"],
  parameters: {
    layout: "padded",
    // Known axe findings, colour contrast on the Badge tints (DESIGN.md,
    // "Story tests"). Back to "error" once they are fixed.
    a11y: { test: "todo" },
    docs: {
      description: {
        component:
          "Kobra's table parts for hand-built tables: `Table` (wrapped in a scrolling, bordered container), `TableHeader`, `TableBody`, `TableFooter`, `TableRow`, `TableHead`, `TableCell` and `TableCaption`. Use `DataTable` for a read-only table driven by a columns array.",
      },
    },
  },
  args: { children: null },
} satisfies Meta<typeof Table>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: (args) => (
    <Table {...args}>
      <TableCaption>Recent invoices</TableCaption>
      <TableHeader>
        <TableRow>
          <TableHead className="w-28">Invoice</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Method</TableHead>
          <TableHead className="text-end">Amount</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {ROWS.map((row) => (
          <TableRow key={row.id}>
            <TableCell className="font-medium">{row.id}</TableCell>
            <TableCell>
              <Badge variant={row.status}>{row.status}</Badge>
            </TableCell>
            <TableCell>{row.method}</TableCell>
            <TableCell className="text-end">{row.amount}</TableCell>
          </TableRow>
        ))}
      </TableBody>
      <TableFooter>
        <TableRow>
          <TableCell colSpan={3}>Total</TableCell>
          <TableCell className="text-end">$750.00</TableCell>
        </TableRow>
      </TableFooter>
    </Table>
  ),
};
