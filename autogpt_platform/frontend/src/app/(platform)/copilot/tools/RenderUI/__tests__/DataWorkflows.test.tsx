import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { showResult } from "./renderConnected";
import {
  dataDashboard,
  formView,
} from "@/lib/openui/__tests__/capability-fixtures";

function submitted(send: ReturnType<typeof vi.fn>) {
  return JSON.parse(
    String(send.mock.calls.at(-1)?.[0]).split("Submitted values:\n")[1],
  );
}
describe("editable data and linked dashboards in chat", () => {
  beforeEach(() => sessionStorage.clear());
  it("updates the same data in charts, totals, filters and submitted values", async () => {
    const send = vi.fn();
    await showResult(send, false, dataDashboard());
    const revenue = await screen.findByLabelText("Revenue · jan");
    expect(await screen.findByText("300 USD")).toBeDefined();
    expect(screen.getByText("Unknown USD")).toBeDefined();
    fireEvent.change(revenue, { target: { value: "125.5" } });
    expect(await screen.findByText("325.5 USD")).toBeDefined();
    fireEvent.click(screen.getByLabelText("Select jan"));
    fireEvent.click(screen.getByLabelText("Only selected grid rows"));
    expect(await screen.findByText("125.5 USD")).toBeDefined();
    expect(screen.getByText("40 USD")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Review changes" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(submitted(send).data.rows[0].cells.revenue).toBe(125.5);
    expect(submitted(send).data.selected).toEqual(["jan"]);
    expect(
      Object.keys(submitted(send)).some((key) => key.endsWith("__draft")),
    ).toBe(false);
  }, 15000);
  it("preserves bad edits and blocks submission even when their row is filtered out", async () => {
    const send = vi.fn();
    await showResult(send, false, dataDashboard());
    fireEvent.change(await screen.findByLabelText("Revenue · jan"), {
      target: { value: "abc" },
    });
    fireEvent.change(screen.getByLabelText("Search Sales data"), {
      target: { value: "February" },
    });
    expect(screen.queryByLabelText("Revenue · jan")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Review changes" }));
    expect(send).not.toHaveBeenCalled();
    expect(screen.getByText(/Correct the highlighted grid/)).toBeDefined();
    fireEvent.change(screen.getByLabelText("Search Sales data"), {
      target: { value: "" },
    });
    expect(
      (screen.getByLabelText("Revenue · jan") as HTMLInputElement).value,
    ).toBe("abc");
    fireEvent.change(screen.getByLabelText("Revenue · jan"), {
      target: { value: "-10.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Review changes" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
  });
  it("keeps row selection through pagination, sorting and row deletion", async () => {
    const rows = Array.from({ length: 15 }, (_, index) => ({
      id: `row${index}`,
      cells: { month: `Month ${index}`, revenue: index, cost: 1 },
    }));
    const send = vi.fn();
    await showResult(send, false, dataDashboard(rows));
    fireEvent.click(await screen.findByLabelText("Select row0"));
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    fireEvent.click(screen.getByLabelText("Select row10"));
    fireEvent.click(screen.getByRole("button", { name: "Remove row10" }));
    fireEvent.click(screen.getByRole("button", { name: "Review changes" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(submitted(send).data.selected).toEqual(["row0"]);
    expect(submitted(send).data.rows).toHaveLength(14);
  });
  it("lets a user choose among five real options, undo, and retain an unknown price", async () => {
    const options = Array.from({ length: 5 }, (_, index) => ({
      id: `hotel${index}`,
      title: `Hotel ${index}`,
      description: "Supplied option",
      facts: [],
      amount: index ? index * 50 : null,
    }));
    const send = vi.fn();
    await showResult(
      send,
      false,
      formView(
        `ResultList("hotel", "Hotels", ${JSON.stringify(options)}, [], "single", true, "USD/night")`,
      ),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Choose Hotel 0" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Choose Hotel 4" }));
    fireEvent.click(screen.getByRole("button", { name: /Undo/ }));
    fireEvent.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(submitted(send).hotel).toEqual(["hotel0"]);
    expect(submitted(send).hotel_amount).toBeNull();
  });
  it("shows generated workspace images inline and exposes recoverable loading errors", async () => {
    await showResult(
      vi.fn(),
      false,
      'root = Workspace("Image result", "", [Artifact("Generated cover", "workspace://generated-file#image/png", "image/png", "cover.png", "A blue mountain illustration")])',
    );
    const image = await screen.findByRole("img", {
      name: "A blue mountain illustration",
    });
    expect(image.getAttribute("src")).toBe(
      "/api/proxy/api/workspace/files/generated-file/download",
    );
    fireEvent.error(image);
    expect(screen.getByText(/This file could not load/)).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Retry preview" }));
    expect(
      screen.getByRole("img", { name: "A blue mountain illustration" }),
    ).toBeDefined();
  });
  it("previews selectable workspace images with retry and protects private images on shares", async () => {
    const options = [
      {
        id: "cover",
        title: "Cover",
        description: "Uploaded image option",
        facts: [],
        image: { url: "workspace://cover-file#image/png", alt: "Blue cover" },
      },
    ];
    const source = formView(
      `ResultList("cover", "Choose a cover", ${JSON.stringify(options)}, [], "single", false, "")`,
    );
    const view = await showResult(vi.fn(), false, source);
    const image = await screen.findByRole("img", { name: "Blue cover" });
    expect(image.getAttribute("src")).toBe(
      "/api/proxy/api/workspace/files/cover-file/download",
    );
    fireEvent.error(image);
    fireEvent.click(screen.getByRole("button", { name: "Retry Blue cover" }));
    expect(screen.getByRole("img", { name: "Blue cover" })).toBeDefined();
    view.unmount();
    await showResult(vi.fn(), true, source);
    expect(screen.queryByRole("img", { name: "Blue cover" })).toBeNull();
    expect(screen.getByText(/image is private/)).toBeDefined();
  });
});

describe("calculated order data", () => {
  it("recomputes row totals, dashboard and submissions after quantity edits", async () => {
    const columns = [
      { key: "item", label: "Item", type: "text" },
      { key: "quantity", label: "Quantity", type: "number" },
      { key: "price", label: "Price", type: "number", unit: "USD" },
      {
        key: "total",
        label: "Total",
        type: "number",
        unit: "USD",
        calculation: {
          operation: "multiply",
          left: "quantity",
          right: "price",
        },
      },
    ];
    const rows = [
      { id: "shirts", cells: { item: "Shirts", quantity: 2, price: 10 } },
      { id: "hats", cells: { item: "Hats", quantity: 3, price: 5 } },
    ];
    const source = `root = Workspace("Order", "", [form, view])
form = Form("order", "Edit", [grid], "Review", "Review my order")
grid = DataGrid("items", "Items", ${JSON.stringify(columns)}, ${JSON.stringify(rows)}, true, "none", [])
view = Dashboard("summary", "Order summary", ${JSON.stringify(columns)}, [], [], [{label:"Order total",column:"total",aggregate:"sum",unit:"USD"}], {form:"order",field:"items"})`;
    const send = vi.fn();
    await showResult(send, false, source);
    expect(await screen.findByText("35 USD")).toBeDefined();
    expect(screen.queryByLabelText("Total · shirts")).toBeNull();
    fireEvent.change(screen.getByLabelText("Quantity · shirts"), {
      target: { value: "3" },
    });
    fireEvent.change(screen.getByLabelText("Quantity · hats"), {
      target: { value: "0" },
    });
    expect(await screen.findByText("30 USD")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Review" }));
    await waitFor(() => expect(send).toHaveBeenCalledOnce());
    expect(submitted(send).items.rows[0].cells.total).toBe(30);
    expect(submitted(send).items.rows[1].cells.total).toBe(0);
  });
});
