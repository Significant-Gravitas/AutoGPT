import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import type { BriefingResponse } from "@/app/api/__generated__/models/briefingResponse";
import type { BriefingRunItem } from "@/app/api/__generated__/models/briefingRunItem";
import { render, screen } from "@/tests/integrations/test-utils";
import { BriefingCard } from "../BriefingCard";

function makeItem(index: number): BriefingRunItem {
  return {
    expert_id: `exp-${index}`,
    expert_name: `Expert ${index}`,
    expert_avatar_url: null,
    agent_name: `Agent ${index}`,
    graph_id: `g-${index}`,
    execution_id: `run-${index}`,
    library_agent_id: `lib-${index}`,
    status: "COMPLETED",
    summary: `Summary ${index}`,
    link: `/library/agents/lib-${index}`,
  };
}

function makeBriefing(items: BriefingRunItem[]): BriefingResponse {
  return {
    id: "briefing-1",
    briefing_date: new Date("2026-08-12T00:00:00Z"),
    created_at: new Date("2026-08-12T00:00:00Z"),
    delivered_at: null,
    content: {
      generated_at: new Date("2026-08-12T00:00:00Z"),
      timezone: "UTC",
      zero_expert_fallback: false,
      run_items: items,
      decision_items: [],
    },
  };
}

const observedTargets: Element[] = [];

beforeEach(() => {
  observedTargets.length = 0;
  vi.stubGlobal(
    "ResizeObserver",
    class {
      observe(target: Element) {
        observedTargets.push(target);
      }
      unobserve() {}
      disconnect() {}
    },
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

test("keeps clipped rows out of the tab order until the list is expanded", async () => {
  const { container } = render(
    <BriefingCard briefing={makeBriefing([0, 1, 2, 3].map(makeItem))} />,
  );

  const clipped = container.querySelectorAll("li")[3];
  expect(clipped.getAttribute("aria-hidden")).toBe("true");
  expect(clipped.querySelector("a")?.getAttribute("tabindex")).toBe("-1");
  expect(screen.queryByRole("link", { name: /Agent 3/ })).toBeNull();

  await userEvent.click(
    screen.getByRole("button", { name: /Show all results \(4\)/ }),
  );

  const expanded = screen.getByRole("link", { name: /Agent 3/ });
  expect(expanded.getAttribute("tabindex")).toBeNull();
  expect(expanded.closest("li")?.getAttribute("aria-hidden")).toBeNull();
});

test("masks agent names and run summaries from session replays", () => {
  render(<BriefingCard briefing={makeBriefing([makeItem(0)])} />);

  expect(screen.getByText("Agent 0").className).not.toContain("sentry-unmask");
  expect(
    screen.getByText(/Summary 0/).className.includes("sentry-unmask"),
  ).toBe(false);
});

test("badges a run that did not complete, and only that run", () => {
  render(
    <BriefingCard
      briefing={makeBriefing([
        makeItem(0),
        { ...makeItem(1), status: "FAILED" },
      ])}
    />,
  );

  const failedRow = screen.getByText("Agent 1").closest("li");
  expect(failedRow?.textContent).toContain("Failed");
  expect(screen.getByText("Agent 0").closest("li")?.textContent).not.toContain(
    "Failed",
  );
});

test("jumps back to the top when the list collapses", async () => {
  const { container } = render(
    <BriefingCard briefing={makeBriefing([0, 1, 2, 3].map(makeItem))} />,
  );
  const list = container.querySelector("ul") as HTMLUListElement;
  const scrollTo = vi.fn();
  list.scrollTo = scrollTo;

  const toggle = screen.getByRole("button", { name: /Show all results/ });
  await userEvent.click(toggle);
  expect(scrollTo).not.toHaveBeenCalled();

  await userEvent.click(screen.getByRole("button", { name: /Show less/ }));
  expect(scrollTo).toHaveBeenCalledWith({ top: 0 });
});

test("re-observes the rows when the briefing refetches a different run list", () => {
  const { rerender } = render(
    <BriefingCard briefing={makeBriefing([0, 1].map(makeItem))} />,
  );
  observedTargets.length = 0;

  rerender(<BriefingCard briefing={makeBriefing([0, 1, 2].map(makeItem))} />);

  expect(
    observedTargets.some((target) => target.textContent?.includes("Agent 2")),
  ).toBe(true);
});
