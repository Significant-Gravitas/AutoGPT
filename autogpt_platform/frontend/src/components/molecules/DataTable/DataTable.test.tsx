import { fireEvent, render, screen, within } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { DataTable } from "./DataTable";
import { getNextSort, sortRows, type DataTableColumn } from "./helpers";

interface Row {
  id: string;
  name: string;
  credits: number | null;
}

const ROWS: Row[] = [
  { id: "1", name: "Beta", credits: 20 },
  { id: "2", name: "alpha", credits: null },
  { id: "3", name: "Gamma", credits: 5 },
];

const COLUMNS: DataTableColumn<Row>[] = [
  {
    key: "name",
    header: "Name",
    cell: (row) => row.name,
    sortValue: (row) => row.name,
  },
  {
    key: "credits",
    header: "Credits",
    align: "right",
    cell: (row) => row.credits ?? "-",
    sortValue: (row) => row.credits,
  },
  { key: "id", header: "ID", cell: (row) => row.id },
];

function renderTable(
  props: Partial<Parameters<typeof DataTable<Row>>[0]> = {},
) {
  return render(
    <DataTable
      columns={COLUMNS}
      rows={ROWS}
      getRowKey={(row) => row.id}
      caption="Agents"
      {...props}
    />,
  );
}

function getColumnText(columnIndex: number) {
  const [, ...bodyRows] = screen.getAllByRole("row");
  return bodyRows.map(
    (row) => within(row).getAllByRole("cell")[columnIndex].textContent,
  );
}

describe("DataTable", () => {
  it("renders the caption, headers and cells", () => {
    renderTable();

    expect(screen.getByRole("table", { name: "Agents" })).toBeDefined();
    expect(screen.getByRole("columnheader", { name: "ID" })).toBeDefined();
    expect(getColumnText(0)).toEqual(["Beta", "alpha", "Gamma"]);
  });

  it("cycles a sortable column through ascending, descending and unsorted", () => {
    renderTable();
    const header = screen.getByRole("columnheader", { name: /Name/ });
    const button = within(header).getByRole("button");

    expect(header.getAttribute("aria-sort")).toBe("none");

    fireEvent.click(button);
    expect(header.getAttribute("aria-sort")).toBe("ascending");
    expect(getColumnText(0)).toEqual(["alpha", "Beta", "Gamma"]);

    fireEvent.click(button);
    expect(header.getAttribute("aria-sort")).toBe("descending");
    expect(getColumnText(0)).toEqual(["Gamma", "Beta", "alpha"]);

    fireEvent.click(button);
    expect(header.getAttribute("aria-sort")).toBe("none");
    expect(getColumnText(0)).toEqual(["Beta", "alpha", "Gamma"]);
  });

  it("leaves non-sortable columns without a sort button", () => {
    renderTable();
    const header = screen.getByRole("columnheader", { name: "ID" });

    expect(header.getAttribute("aria-sort")).toBeNull();
    expect(within(header).queryByRole("button")).toBeNull();
  });

  it("reports sort changes without reordering when sorting is manual", () => {
    const onSortChange = vi.fn();
    renderTable({ manualSorting: true, sort: null, onSortChange });

    fireEvent.click(
      within(screen.getByRole("columnheader", { name: /Name/ })).getByRole(
        "button",
      ),
    );

    expect(onSortChange).toHaveBeenCalledWith({
      key: "name",
      direction: "asc",
    });
    expect(getColumnText(0)).toEqual(["Beta", "alpha", "Gamma"]);
  });

  it("renders skeleton rows while loading", () => {
    renderTable({ isLoading: true, loadingRowCount: 2 });

    expect(screen.getByRole("table").getAttribute("aria-busy")).toBe("true");
    expect(screen.getAllByRole("row")).toHaveLength(3);
    expect(screen.queryByText("Beta")).toBeNull();
  });

  it("renders the empty state across every column", () => {
    renderTable({ rows: [], emptyState: "No agents yet" });

    const cell = screen.getByRole("cell", { name: "No agents yet" });
    expect(cell.getAttribute("colspan")).toBe("3");
  });

  it("calls onRowClick on click, Enter and Space", () => {
    const onRowClick = vi.fn();
    renderTable({ onRowClick });
    const [, firstRow] = screen.getAllByRole("row");

    fireEvent.click(firstRow);
    fireEvent.keyDown(firstRow, { key: "Enter" });
    fireEvent.keyDown(firstRow, { key: " " });

    expect(onRowClick).toHaveBeenCalledTimes(3);
    expect(onRowClick).toHaveBeenCalledWith(ROWS[0]);
    expect(firstRow.getAttribute("tabindex")).toBe("0");
  });

  it("does not make rows focusable without onRowClick and forwards its ref", () => {
    const ref = createRef<HTMLTableElement>();
    renderTable({ ref });
    const [, firstRow] = screen.getAllByRole("row");

    expect(firstRow.getAttribute("tabindex")).toBeNull();
    expect(ref.current).toBe(screen.getByRole("table"));
  });
});

describe("DataTable helpers", () => {
  it("sorts numbers and keeps empty values last in both directions", () => {
    const asc = sortRows(ROWS, COLUMNS, { key: "credits", direction: "asc" });
    const desc = sortRows(ROWS, COLUMNS, { key: "credits", direction: "desc" });

    expect(asc.map((row) => row.credits)).toEqual([5, 20, null]);
    expect(desc.map((row) => row.credits)).toEqual([20, 5, null]);
  });

  it("starts a new column ascending", () => {
    expect(getNextSort({ key: "name", direction: "desc" }, "credits")).toEqual({
      key: "credits",
      direction: "asc",
    });
  });
});
