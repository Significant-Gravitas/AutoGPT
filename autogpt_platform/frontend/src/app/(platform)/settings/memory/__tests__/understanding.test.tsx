import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import {
  getListExpertsMockHandler200,
  getListExpertsResponseMock200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getForgetMyBusinessUnderstandingMockHandler200,
  getGetMyBusinessUnderstandingMockHandler200,
  getGetMyExpertMemoryOverviewMockHandler200,
  getGetMyMemoryOverviewMockHandler200,
  getListMyExpertMemoryFactsMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import type { MemoryUnderstanding } from "@/app/api/__generated__/models/memoryUnderstanding";
import { server } from "@/mocks/mock-server";

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

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    DREAM_SKILL_LEARNING_ENABLED: "dream-skill-learning-enabled",
    CHAT_MODE_OPTION: "chat-mode-option",
    ENABLE_PLATFORM_PAYMENT: "enable-platform-payment",
  },
  useGetFlag: () => false,
  useFlagStatus: () => ({ enabled: false, answered: true }),
}));

import SettingsMemoryPage from "../page";

const LONG_NOTE = `Onboarding brain dump (typed): ${"We repair and rent bikes. ".repeat(20)}`;

const KEPT: MemoryUnderstanding = {
  updated_at: new Date("2026-10-22T10:00:00Z"),
  user_name: "Sam",
  business_name: "Harbor Bikes",
  industry: null,
  pain_points: ["Chasing invoices", "Booking repairs by phone"],
  current_software: [],
  additional_notes: LONG_NOTE,
};

const NOTHING_KEPT: MemoryUnderstanding = {
  updated_at: null,
  pain_points: [],
  current_software: [],
};

function mockAccountScope(understanding: () => MemoryUnderstanding) {
  server.use(
    getListExpertsMockHandler200([]),
    getListMyMemoryFactsMockHandler200({ expert_id: null, items: [] }),
    getGetMyMemoryOverviewMockHandler200({
      expert_id: null,
      facts: 0,
      entities: 0,
      episodes: 0,
    }),
    getGetMyBusinessUnderstandingMockHandler200(() => understanding()),
  );
}

beforeEach(() => {
  resetCopilotChatRegistry();
  window.history.replaceState({}, "", "/settings/memory");
});

describe("What Otto knows about your work", () => {
  it("lists what onboarding stored, before any chat", async () => {
    mockAccountScope(() => KEPT);
    render(<SettingsMemoryPage />);

    expect(
      await screen.findByRole("heading", {
        name: "What Otto knows about your work",
      }),
    ).toBeDefined();
    expect(screen.getByText("Harbor Bikes")).toBeDefined();
    expect(screen.getByText("Booking repairs by phone")).toBeDefined();
    expect(screen.queryByText("Industry")).toBeNull();
    expect(screen.queryByText("Tools you use")).toBeNull();
    expect(screen.getByText(/Nothing remembered yet/)).toBeDefined();
  });

  it("shortens a long brain dump until asked", async () => {
    mockAccountScope(() => KEPT);
    render(<SettingsMemoryPage />);

    const showAll = await screen.findByRole("button", { name: "Show all" });
    expect(screen.queryByText(LONG_NOTE.trim())).toBeNull();

    await userEvent.setup().click(showAll);

    expect(
      await screen.findByText(
        (_, node) =>
          node?.tagName === "DD" &&
          node.textContent === `${LONG_NOTE}Show less`,
      ),
    ).toBeDefined();
  });

  it("keeps an emoji whole where the preview is cut", async () => {
    const note = `${"a".repeat(279)}😀 and more after the cut`;
    mockAccountScope(() => ({ ...KEPT, additional_notes: note }));
    render(<SettingsMemoryPage />);

    await screen.findByRole("button", { name: "Show all" });

    expect(
      screen.getByText(
        (_, node) =>
          node?.tagName === "DD" &&
          node.textContent === `${"a".repeat(279)}😀…Show all`,
      ),
    ).toBeDefined();
  });

  it("shows no card when nothing is kept", async () => {
    let served = 0;
    mockAccountScope(() => {
      served += 1;
      return NOTHING_KEPT;
    });
    render(<SettingsMemoryPage />);

    await screen.findByText(/Nothing remembered yet/);
    await waitFor(() => expect(served).toBe(1));
    await settle();
    expect(screen.queryByText("What Otto knows about your work")).toBeNull();
  });

  it("forgets it after a confirmation", async () => {
    let kept = KEPT;
    let forgetCalls = 0;
    mockAccountScope(() => kept);
    server.use(
      getForgetMyBusinessUnderstandingMockHandler200(() => {
        forgetCalls += 1;
        kept = NOTHING_KEPT;
        return { forgotten: true };
      }),
    );
    render(<SettingsMemoryPage />);
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: "Forget this" }),
    );
    const confirm = await screen.findByRole("button", { name: "Forget" });
    expect(forgetCalls).toBe(0);
    await user.click(confirm);

    await waitFor(() => expect(forgetCalls).toBe(1));
    await waitFor(() => expect(screen.queryByText("Harbor Bikes")).toBeNull(), {
      timeout: 3000,
    });
  });

  it("is not shown in an expert's scope", async () => {
    const maria = {
      ...getListExpertsResponseMock200()[0],
      id: "expert-maria",
      name: "Maria",
      avatar_url: null,
      is_archived: false,
    };
    server.use(
      getListExpertsMockHandler200([maria]),
      getGetMyExpertMemoryOverviewMockHandler200({
        expert_id: "expert-maria",
        facts: 1,
        entities: 1,
        episodes: 1,
      }),
      getListMyExpertMemoryFactsMockHandler200({
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
      }),
      getGetMyBusinessUnderstandingMockHandler200(KEPT),
    );
    window.history.replaceState({}, "", "/settings/memory?expert=expert-maria");
    render(<SettingsMemoryPage />);

    await screen.findByText("Q4 campaign brief is due Friday");
    await settle();
    expect(screen.queryByText("What Otto knows about your work")).toBeNull();
  });
});

// Long enough for a fetched understanding to render if the page asked for one.
function settle() {
  return new Promise((resolve) => setTimeout(resolve, 300));
}
