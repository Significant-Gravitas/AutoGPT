import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getEraseMyMemoryMockHandler200,
  getForgetMyMemoryFactMockHandler200,
  getGetMyMemoryOverviewMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import { server } from "@/mocks/mock-server";

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    GRAPHITI_MEMORY: "graphiti-memory",
    HIRE_EXPERTS: "hire-experts",
  },
  useGetFlag: () => true,
}));

vi.mock("@/services/feature-flags/with-feature-flag", () => ({
  withFeatureFlag: (Component: React.ComponentType) => Component,
}));

import SettingsMemoryPage from "../page";

const FACTS = {
  expert_id: null,
  items: [
    {
      uuid: "edge-1",
      fact: "Runs a DTC candle brand called Emberline",
      name: "runs",
      source: "User",
      target: "Emberline",
      created_at: "2026-08-16T00:00:00Z",
    },
    {
      uuid: "edge-2",
      fact: "Prefers weekly summary emails on Monday mornings",
      name: "prefers",
      source: "User",
      target: "Monday summaries",
      created_at: "2026-08-14T00:00:00Z",
    },
  ],
};

function mockHappyPath() {
  server.use(
    getListExpertsMockHandler200([]),
    getListMyMemoryFactsMockHandler200(FACTS),
    getGetMyMemoryOverviewMockHandler200({
      expert_id: null,
      facts: 214,
      entities: 90,
      episodes: 41,
    }),
  );
}

describe("Settings memory page", () => {
  it("renders recent memories for the AutoPilot scope", async () => {
    mockHappyPath();
    render(<SettingsMemoryPage />);

    expect(
      await screen.findByText("Runs a DTC candle brand called Emberline"),
    ).toBeDefined();
    expect(
      screen.getByText("Prefers weekly summary emails on Monday mornings"),
    ).toBeDefined();
    expect(screen.getByRole("heading", { name: "Memory" })).toBeDefined();

    const summaryLink = screen.getByRole("link", { name: "View my summary" });
    expect(summaryLink.getAttribute("href")).toBe(
      "/copilot?seed=memory-summary",
    );
    const topicLink = screen.getByRole("link", { name: "Forget a topic…" });
    expect(topicLink.getAttribute("href")).toBe("/copilot?seed=memory-forget");
  });

  it("shows the empty state when nothing is remembered yet", async () => {
    server.use(
      getListExpertsMockHandler200([]),
      getListMyMemoryFactsMockHandler200({ expert_id: null, items: [] }),
      getGetMyMemoryOverviewMockHandler200({
        expert_id: null,
        facts: 0,
        entities: 0,
        episodes: 0,
      }),
    );
    render(<SettingsMemoryPage />);

    expect(await screen.findByText(/Nothing remembered yet/)).toBeDefined();
  });

  it("forgets a single fact", async () => {
    mockHappyPath();
    const forgotten: string[] = [];
    server.use(
      getForgetMyMemoryFactMockHandler200((info) => {
        forgotten.push(String(info.params.factUuid));
        return { uuid: String(info.params.factUuid), forgotten: true };
      }),
    );
    render(<SettingsMemoryPage />);

    await screen.findByText("Runs a DTC candle brand called Emberline");
    const user = userEvent.setup();
    await user.click(screen.getAllByRole("button", { name: "Forget" })[0]);

    await waitFor(() => expect(forgotten).toEqual(["edge-1"]));
  });

  it("gates the scope erase behind typed confirmation", async () => {
    mockHappyPath();
    let erased = false;
    server.use(
      getEraseMyMemoryMockHandler200(() => {
        erased = true;
        return { expert_id: null, deleted_nodes: 214, erased: true };
      }),
    );
    render(<SettingsMemoryPage />);

    await screen.findByText("Runs a DTC candle brand called Emberline");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Erase memory" }));

    const confirm = await screen.findByRole("button", {
      name: "Erase everything",
    });
    expect(confirm.hasAttribute("disabled")).toBe(true);
    expect(screen.getByText(/214 memories/)).toBeDefined();

    await user.type(screen.getByPlaceholderText("AutoPilot"), "AutoPilot");
    await waitFor(() => expect(confirm.hasAttribute("disabled")).toBe(false));

    await user.click(confirm);
    await waitFor(() => expect(erased).toBe(true));
  });
});
