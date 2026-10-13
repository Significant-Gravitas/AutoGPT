import { screen, within } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { workspaceText } from "@/lib/openui/parse";
import { connected } from "@/lib/openui/__tests__/connected-fixtures";
import { revisedLaunch } from "@/lib/openui/__tests__/schedule-fixtures";
import { showResult } from "./renderConnected";

function selectedTrip(choice: string, nights = 1) {
  return connected
    .replace(
      '"Where would you stay?", ""',
      `"Where would you stay?", "${choice}"`,
    )
    .replace('"Nights", 1,', `"Nights", ${nights},`);
}

describe("saved calculations agree with the interactive response", () => {
  beforeEach(() => sessionStorage.clear());

  it.each([
    { choice: "city", title: "City hotel", nights: 1, expected: "$267.74" },
    { choice: "city", title: "City hotel", nights: 3, expected: "$646.72" },
    { choice: "coast", title: "Coastal cabin", nights: 1, expected: "$227.70" },
    { choice: "coast", title: "Coastal cabin", nights: 3, expected: "$526.60" },
  ])(
    "keeps shared nightly charges and flat fees accurate for $choice / $nights nights",
    async ({ choice, title, nights, expected }) => {
      const source = selectedTrip(choice, nights)
        .replace("amount:180", "amount:179.99")
        .replace("amount:140", "amount:139.95")
        .replace(
          "stay_amount * nights + extras_total",
          "(stay_amount + 9.50) * nights + extras_total + 12.25",
        )
        .replace('"USD", 2)', '"USD", 0)');
      await showResult(vi.fn(), false, source);
      const metric = within(
        screen.getByRole("region", { name: "Estimated total" }),
      );
      expect(await metric.findByText(expected)).toBeDefined();
      expect(workspaceText(source)).toContain(`Estimated total: ${expected}`);
      expect(workspaceText(source)).toContain(`${title} (selected)`);
    },
  );

  it("does not invent a price or a selection before the user chooses", async () => {
    await showResult();
    const summary = workspaceText(connected);
    expect(summary).toContain("Estimated total: Unknown / incomplete inputs");
    expect(summary).not.toContain("(selected)");
    expect(
      await screen.findByText("Complete your choices to calculate this total."),
    ).toBeDefined();
    expect(screen.queryByText("$246.00")).toBeNull();
  });

  it.each([
    { row: "quantity:2,unitPrice:null", description: "2 × unknown price" },
    {
      row: "quantity:null,unitPrice:25",
      description: "unknown quantity × $25.00",
    },
  ])(
    "keeps an included unknown cost incomplete: $description",
    async ({ row, description }) => {
      const source = selectedTrip("city").replace(
        "quantity:2,unitPrice:25",
        row,
      );
      await showResult(vi.fn(), false, source);
      const summary = workspaceText(source);
      expect(summary).toContain(`Museum tickets: ${description} = Unknown`);
      expect(summary).toContain("Subtotal: Unknown");
      expect(summary).toContain("Estimated total: Unknown / incomplete inputs");
      expect(
        await screen.findByText(
          "Check the highlighted inputs to calculate this total.",
        ),
      ).toBeDefined();
      expect(screen.queryByText("$196.00")).toBeNull();
    },
  );

  it("calculates after an unknown row is explicitly excluded without calling it free", async () => {
    const source = selectedTrip("city").replace(
      "quantity:2,unitPrice:25,included:true",
      "quantity:2,unitPrice:null,included:false",
    );
    await showResult(vi.fn(), false, source);
    expect(await screen.findByText("$196.00")).toBeDefined();
    const summary = workspaceText(source);
    expect(summary).toContain("Museum tickets: 2 × unknown price (excluded)");
    expect(summary).toContain("Subtotal: $16.00");
    expect(summary).toContain("Estimated total: $196.00");
    expect(summary).not.toContain("$0.00");
  });

  it("pauses a division by zero in both outputs without losing the rest of the plan", async () => {
    const source = selectedTrip("city").replace(
      "stay_amount * nights + extras_total",
      "stay_amount / (nights - 1)",
    );
    await showResult(vi.fn(), false, source);
    expect(await screen.findByText("Cannot divide by zero.")).toBeDefined();
    const summary = workspaceText(source);
    expect(summary).toContain("Estimated total: Unknown / incomplete inputs");
    expect(summary).toContain("City hotel (selected)");
    expect(summary).not.toMatch(/NaN|Infinity/);
  });

  it("preserves completed work and an infeasible revised schedule in the saved summary", async () => {
    const source = revisedLaunch
      .replace("[form])", "[form, remaining])")
      .concat(
        '\nremaining = CalculatedMetric("launch", "Remaining work", "work_required_minutes", "number", "min", 0)',
      );
    await showResult(vi.fn(), false, source);
    expect(await screen.findByText("60 min over available time")).toBeDefined();
    expect(screen.getByText("60 min unscheduled · 0 min unused")).toBeDefined();
    expect(await screen.findByText("660 min")).toBeDefined();
    const summary = workspaceText(source);
    expect(summary).toContain(
      "Research: completed; excluded from remaining work",
    );
    expect(summary).toContain("Monday available: 14:00–17:00");
    expect(summary).toContain(
      "660 min remaining; 600 min available; 60 min unscheduled; plan does not fit.",
    );
    expect(summary).toContain("60 min over available time.");
    expect(summary).toContain("Remaining work: 660 min");
  });
});
