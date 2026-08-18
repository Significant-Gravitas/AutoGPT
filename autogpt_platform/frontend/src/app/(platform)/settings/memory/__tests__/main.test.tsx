import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import {
  getListExpertsMockHandler200,
  getListExpertsResponseMock200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2GetSessionMockHandler200,
  getGetV2GetSessionResponseMock200,
  getPostV2CreateSessionMockHandler200,
  getPostV2CreateSessionResponseMock200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getEraseMyMemoryMockHandler200,
  getForgetMyMemoryFactMockHandler200,
  getGetMyExpertMemoryOverviewMockHandler200,
  getGetMyMemoryOverviewMockHandler200,
  getListMyExpertMemoryFactsMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import { server } from "@/mocks/mock-server";
import {
  assistantTextChunks,
  streamSseResponse,
} from "@/tests/integrations/copilot-sse";

// The CoPilot stream transport talks to the backend host directly (it
// bypasses the Next proxy), so MSW must match an absolute URL.
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

vi.mock("@/app/(platform)/copilot/helpers", async (importActual) => {
  const actual =
    await importActual<typeof import("@/app/(platform)/copilot/helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    GRAPHITI_MEMORY: "graphiti-memory",
    HIRE_EXPERTS: "hire-experts",
    ARTIFACTS: "artifacts",
    CHAT_MODE_OPTION: "chat-mode-option",
    ENABLE_PLATFORM_PAYMENT: "enable-platform-payment",
  },
  useGetFlag: (flag: string) =>
    flag === "graphiti-memory" || flag === "hire-experts",
}));

vi.mock("@/services/feature-flags/with-feature-flag", () => ({
  withFeatureFlag: (Component: React.ComponentType) => Component,
}));

import SettingsMemoryPage from "../page";

const SESSION_ID = "memory-chat-session-1";

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

beforeEach(() => {
  resetCopilotChatRegistry();
});

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
    expect(
      screen.getByRole("button", { name: "View my summary" }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Forget a topic…" }),
    ).toBeDefined();
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

  it("switches scope to an expert through the scope selector", async () => {
    const maria = {
      ...getListExpertsResponseMock200()[0],
      id: "expert-maria",
      name: "Maria",
      role: "Growth Marketer",
      avatar_url: null,
      is_archived: false,
    };
    const expertFactRequests: string[] = [];
    server.use(
      getListExpertsMockHandler200([maria]),
      getListMyMemoryFactsMockHandler200(FACTS),
      getGetMyMemoryOverviewMockHandler200({
        expert_id: null,
        facts: 214,
        entities: 90,
        episodes: 41,
      }),
      getListMyExpertMemoryFactsMockHandler200((info) => {
        expertFactRequests.push(String(info.params.expertId));
        return {
          expert_id: "expert-maria",
          items: [
            {
              uuid: "edge-m1",
              fact: "Q4 campaign brief is due Friday",
              name: "due",
              source: "Campaign",
              target: "Friday",
              created_at: "2026-08-17T00:00:00Z",
            },
          ],
        };
      }),
      getGetMyExpertMemoryOverviewMockHandler200({
        expert_id: "expert-maria",
        facts: 12,
        entities: 8,
        episodes: 3,
      }),
    );
    render(<SettingsMemoryPage />);

    await screen.findByText("Runs a DTC candle brand called Emberline");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Memory scope" }));
    await user.click(
      await screen.findByRole("menuitem", { name: /Growth Marketer/ }),
    );

    expect(
      await screen.findByText("Q4 campaign brief is due Friday"),
    ).toBeDefined();
    expect(expertFactRequests).toEqual(["expert-maria"]);
    expect(
      screen.getByRole("button", { name: "View Maria's summary" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Erase memory" })).toBeDefined();
    expect(screen.getByText(/Erase Maria's memory/)).toBeDefined();
  });

  it("opens the summary chat in-pane and auto-sends the seeded prompt", async () => {
    mockHappyPath();
    const createBodies: unknown[] = [];
    const streamBodies: string[] = [];
    server.use(
      getPostV2CreateSessionMockHandler200(async (info) => {
        createBodies.push(await info.request.clone().json());
        return getPostV2CreateSessionResponseMock200({ id: SESSION_ID });
      }),
      getGetV2GetSessionMockHandler200(
        getGetV2GetSessionResponseMock200({
          id: SESSION_ID,
          messages: [],
          active_stream: null,
        }),
      ),
      http.post(
        `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${SESSION_ID}/stream`,
        async ({ request }) => {
          streamBodies.push(await request.clone().text());
          return streamSseResponse(
            assistantTextChunks("Here's what I know about you."),
            { abortSignal: request.signal },
          );
        },
      ),
    );
    render(<SettingsMemoryPage />);

    await screen.findByText("Runs a DTC candle brand called Emberline");
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "View my summary" }));

    expect(await screen.findByText("AutoPilot's memory")).toBeDefined();
    await waitFor(() => expect(createBodies.length).toBe(1));
    await waitFor(() => expect(streamBodies.length).toBe(1));
    expect(streamBodies[0]).toContain(
      "Give me a summary of everything you know about me",
    );
    expect(
      await screen.findByText("Here's what I know about you."),
    ).toBeDefined();
  });
});
