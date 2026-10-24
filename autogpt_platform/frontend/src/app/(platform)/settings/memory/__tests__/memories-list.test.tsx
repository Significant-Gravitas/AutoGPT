import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import {
  getListExpertsMockHandler200,
  getListExpertsResponseMock200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getForgetMyMemoryFactMockHandler200,
  getGetMyBusinessUnderstandingMockHandler200,
  getGetMyExpertMemoryOverviewMockHandler200,
  getGetMyMemoryOverviewMockHandler200,
  getListMyExpertMemoryFactsMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import type { MemoryFact } from "@/app/api/__generated__/models/memoryFact";
import { server } from "@/mocks/mock-server";

const { toast } = vi.hoisted(() => ({ toast: vi.fn() }));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast,
}));

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => TEST_BACKEND_BASE_URL,
    },
  };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

import SettingsMemoryPage from "../page";

function makeFacts(count: number, from = 0): MemoryFact[] {
  return Array.from({ length: count }, (_, i) => ({
    uuid: `edge-${from + i}`,
    fact: `Memory number ${from + i}`,
    name: "knows",
    source: "User",
    target: `Thing ${from + i}`,
    created_at: "2026-10-23T12:00:00Z",
  }));
}

function mockFacts(facts: MemoryFact[]) {
  server.use(
    getListMyMemoryFactsMockHandler200({
      expert_id: null,
      items: facts,
      total: facts.length,
      next_cursor: null,
    }),
  );
}

function rows() {
  return screen
    .queryAllByText(/^Memory number \d+$/)
    .map((el) => el.textContent);
}

beforeEach(() => {
  toast.mockClear();
  resetCopilotChatRegistry();
  window.history.replaceState({}, "", "/settings/memory");
  server.use(
    getListExpertsMockHandler200([]),
    getGetMyBusinessUnderstandingMockHandler200(),
    getGetMyMemoryOverviewMockHandler200({
      expert_id: null,
      facts: 19,
      entities: 20,
      episodes: 4,
    }),
  );
});

describe("Settings memory page — the memories list", () => {
  it("shows how many memories there are and lists every one on Show all", async () => {
    mockFacts(makeFacts(19));
    render(<SettingsMemoryPage />);

    expect(await screen.findByText("19 memories")).toBeDefined();
    expect(rows()).toHaveLength(6);

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Show all 19" }));

    expect(rows()).toEqual(makeFacts(19).map((fact) => fact.fact));
    expect(screen.getByText("All memories")).toBeDefined();

    await user.click(screen.getByRole("button", { name: "Show fewer" }));
    expect(rows()).toHaveLength(6);
  });

  it("opens another scope on its recent memories, not the expanded list", async () => {
    const maria = {
      ...getListExpertsResponseMock200()[0],
      id: "expert-maria",
      name: "Maria",
      role: "Growth Marketer",
      avatar_url: null,
      is_archived: false,
    };
    mockFacts(makeFacts(8));
    server.use(
      getListExpertsMockHandler200([maria]),
      getListMyExpertMemoryFactsMockHandler200({
        expert_id: "expert-maria",
        items: makeFacts(9, 100),
        total: 9,
      }),
      getGetMyExpertMemoryOverviewMockHandler200({
        expert_id: "expert-maria",
        facts: 9,
        entities: 3,
        episodes: 1,
      }),
    );
    render(<SettingsMemoryPage />);

    expect(await screen.findByText("8 memories")).toBeDefined();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Show all 8" }));
    expect(rows()).toHaveLength(8);

    await user.click(screen.getByRole("button", { name: "Memory scope" }));
    await user.click(
      await screen.findByRole("menuitem", { name: /Growth Marketer/ }),
    );

    expect(await screen.findByText("9 memories")).toBeDefined();
    expect(screen.getByText("Recent memories")).toBeDefined();
    expect(rows()).toHaveLength(6);
  });

  it("loads the next page with the cursor the last page handed out", async () => {
    const cursors: (string | null)[] = [];
    server.use(
      getListMyMemoryFactsMockHandler200(({ request }) => {
        const cursor = new URL(request.url).searchParams.get("cursor");
        cursors.push(cursor);
        return cursor === "page-2"
          ? { expert_id: null, items: makeFacts(3, 50), total: 53 }
          : {
              expert_id: null,
              items: makeFacts(50),
              total: 53,
              next_cursor: "page-2",
            };
      }),
    );
    render(<SettingsMemoryPage />);

    expect(await screen.findByText("53 memories")).toBeDefined();
    const user = userEvent.setup();
    expect(screen.queryByRole("button", { name: "Show all 53" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Show all" }));
    expect(rows()).toHaveLength(50);

    await user.click(screen.getByRole("button", { name: "Show more" }));

    await waitFor(() => expect(rows()).toHaveLength(53));
    expect(rows().at(-1)).toBe("Memory number 52");
    expect(cursors).toEqual([null, "page-2"]);
    expect(screen.queryByRole("button", { name: "Show more" })).toBeNull();
  });

  it("keeps the loaded memories and says so when Show more fails", async () => {
    server.use(
      http.get(/\/api\/proxy\/api\/memory\/facts(\?.*)?$/, ({ request }) =>
        new URL(request.url).searchParams.get("cursor")
          ? HttpResponse.json({ detail: "Memory is down" }, { status: 500 })
          : HttpResponse.json({
              expert_id: null,
              items: makeFacts(50),
              total: 53,
              next_cursor: "page-2",
            }),
      ),
    );
    render(<SettingsMemoryPage />);

    expect(await screen.findByText("53 memories")).toBeDefined();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Show all" }));
    await user.click(screen.getByRole("button", { name: "Show more" }));

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Could not load more memories",
          variant: "destructive",
        }),
      ),
    );
    expect(rows()).toHaveLength(50);
    expect(screen.getByText("All memories")).toBeDefined();
    expect(screen.queryByText(/when retrieving memory/)).toBeNull();
  });

  it("removes a forgotten memory at once, with no row left waiting on the server", async () => {
    mockFacts(makeFacts(8));
    let finishForget: () => void = () => {};
    const forgetDone = new Promise<void>((resolve) => {
      finishForget = resolve;
    });
    const forgotten: string[] = [];
    server.use(
      getForgetMyMemoryFactMockHandler200(async (info) => {
        forgotten.push(String(info.params.factUuid));
        await forgetDone;
        return { uuid: String(info.params.factUuid), forgotten: true };
      }),
    );
    render(<SettingsMemoryPage />);

    await screen.findByText("Memory number 0");
    const user = userEvent.setup();
    await user.click(screen.getAllByRole("button", { name: "Forget" })[0]);

    expect(screen.queryByText("Memory number 0")).toBeNull();
    expect(screen.getByText("7 memories")).toBeDefined();
    expect(rows()).toHaveLength(6);
    expect(
      screen
        .getAllByRole("button", { name: "Forget" })
        .every((button) => !button.hasAttribute("disabled")),
    ).toBe(true);
    await waitFor(() => expect(forgotten).toEqual(["edge-0"]));

    finishForget();
    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith({ title: "Forgotten" }),
    );
    expect(screen.queryByText("Memory number 0")).toBeNull();
  });

  it("puts a memory back and says so when the forget fails", async () => {
    mockFacts(makeFacts(3));
    server.use(
      http.delete(/\/api\/proxy\/api\/memory\/facts\/[^/]+$/, () =>
        HttpResponse.json(
          { detail: "Could not forget this memory" },
          { status: 500 },
        ),
      ),
    );
    render(<SettingsMemoryPage />);

    await screen.findByText("Memory number 0");
    const user = userEvent.setup();
    await user.click(screen.getAllByRole("button", { name: "Forget" })[0]);

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Could not forget that memory",
          variant: "destructive",
        }),
      ),
    );
    expect(screen.getByText("Memory number 0")).toBeDefined();
    expect(screen.getByText("3 memories")).toBeDefined();
  });

  it("drops a memory that was already forgotten elsewhere and refreshes the list", async () => {
    let listCalls = 0;
    server.use(
      getListMyMemoryFactsMockHandler200(() => {
        listCalls += 1;
        const facts = listCalls === 1 ? makeFacts(3) : makeFacts(2, 1);
        return { expert_id: null, items: facts, total: facts.length };
      }),
      http.delete(/\/api\/proxy\/api\/memory\/facts\/[^/]+$/, () =>
        HttpResponse.json({ detail: "Memory not found" }, { status: 404 }),
      ),
    );
    render(<SettingsMemoryPage />);

    await screen.findByText("Memory number 0");
    const user = userEvent.setup();
    await user.click(screen.getAllByRole("button", { name: "Forget" })[0]);

    await waitFor(() => expect(listCalls).toBe(2));
    expect(screen.queryByText("Memory number 0")).toBeNull();
    expect(await screen.findByText("2 memories")).toBeDefined();
    expect(toast).toHaveBeenCalledWith({ title: "Already forgotten" });
  });

  it("opens the erase confirmation from Forget everything", async () => {
    mockFacts(makeFacts(3));
    render(<SettingsMemoryPage />);

    await screen.findByText("Memory number 0");
    const user = userEvent.setup();
    await user.click(
      screen.getByRole("button", { name: "Forget everything…" }),
    );

    const confirm = await screen.findByRole("button", {
      name: "Erase everything",
    });
    expect(confirm.hasAttribute("disabled")).toBe(true);
    expect(screen.getByPlaceholderText("Otto")).toBeDefined();
  });
});
