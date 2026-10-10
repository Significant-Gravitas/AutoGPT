import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { resetCopilotChatRegistry } from "@/app/(platform)/copilot/copilotChatRegistry";
import { TEST_BACKEND_BASE_URL } from "@/app/(platform)/copilot/__tests__/sse-helpers";
import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetMyMemoryOverviewMockHandler200,
  getListMyMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import {
  getDecideSkillLearningProposalMockHandler200,
  getGetSkillLearningDetailMockHandler200,
  getListSkillLearningDecisionsMockHandler200,
  getListSkillLearningHistoryMockHandler200,
  getRestoreLearnedSkillVersionMockHandler200,
} from "@/app/api/__generated__/endpoints/skill-learning/skill-learning.msw";
import { SkillLearningDetail } from "@/app/api/__generated__/models/skillLearningDetail";
import { within } from "@testing-library/react";
import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
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

vi.mock("@/app/(platform)/copilot/helpers", async (importActual) => {
  const actual =
    await importActual<typeof import("@/app/(platform)/copilot/helpers")>();
  return {
    ...actual,
    getCopilotAuthHeaders: async () => ({ "x-test-auth": "yes" }),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "graphiti-memory" ||
      flag === "hire-experts" ||
      flag === "dream-skill-learning-enabled",
  };
});

vi.mock("@/services/feature-flags/with-feature-flag", () => ({
  withFeatureFlag: (Component: React.ComponentType) => Component,
}));

import SettingsMemoryPage from "../page";

const proposal: SkillVersionSummary = {
  id: "ver-9",
  skill_name: "csv-import-checks",
  expert_id: "expert-1",
  version: 4,
  origin: "saved_overnight",
  origin_label: "Saved overnight",
  summary: "Added a duplicate column check",
  state: "needs_decision",
  state_label: "Needs your decision",
  state_reason: "automatic improvements are off for this skill",
  created_at: new Date("2026-09-14T03:00:00Z"),
  description: "Import a CSV",
  triggers: [],
  body: "## Steps\n1. Reject duplicate columns\n",
  evidence: [],
  limits: [],
  sources: [],
  use: {
    version_id: "ver-9",
    loads: 0,
    checks_passed: 0,
    checks_failed: 0,
    reported_working: 0,
    reported_failed: 0,
    stopped_mid_use: 0,
    unknown: 0,
    reuse_label: "Not yet reused",
  },
  base_version_id: "ver-3",
  restored_from_version_id: null,
  blocked_pattern_class: null,
  blocked_step: null,
};

beforeEach(() => {
  resetCopilotChatRegistry();
  window.history.replaceState({}, "", "/settings/memory");
  server.use(
    getListExpertsMockHandler200([]),
    getListMyMemoryFactsMockHandler200({ expert_id: null, items: [] }),
    getGetMyMemoryOverviewMockHandler200({
      expert_id: null,
      facts: 0,
      entities: 0,
      episodes: 0,
    }),
    getListSkillLearningHistoryMockHandler200({
      expert_id: null,
      items: [
        {
          id: "ver-3",
          kind: "version",
          skill_name: "csv-import-checks",
          expert_id: "expert-1",
          created_at: new Date("2026-09-13T03:00:00Z"),
          summary: "Added an encoding check after the previous import failed.",
          origin: "saved_overnight",
          origin_label: "Saved overnight",
          state: "ready",
          state_label: "Ready to use",
          version: 3,
          version_id: "ver-3",
          source: null,
        },
        {
          id: "rev-1",
          kind: "review",
          skill_name: null,
          expert_id: "expert-1",
          created_at: new Date("2026-09-12T03:00:00Z"),
          summary:
            "no concrete outcome (tool check or explicit confirmation) recorded",
          origin: null,
          origin_label: null,
          state: "skipped",
          state_label: "Skipped",
          version: null,
          version_id: null,
          source: null,
        },
      ],
    }),
    getListSkillLearningDecisionsMockHandler200({ items: [proposal] }),
  );
});

describe("Settings memory page — learning", () => {
  it("shows learning history with plain explanations and honest skips", async () => {
    render(<SettingsMemoryPage />);

    expect(await screen.findByText("Learning history")).toBeDefined();
    expect(
      await screen.findByText(
        "Added an encoding check after the previous import failed.",
      ),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "csv-import-checks v3" }),
    ).toBeDefined();
    expect(
      screen.getByText(
        "no concrete outcome (tool check or explicit confirmation) recorded",
      ),
    ).toBeDefined();
  });

  it("lists open decisions and records a keep-current decision", async () => {
    const decide = vi.fn();
    server.use(
      getDecideSkillLearningProposalMockHandler200(
        async ({ request, params }) => {
          decide({ params, body: await request.json() });
          return {
            status: "applied",
            status_label: "Applied",
            reason: "kept current",
            version: null,
            pattern_class: null,
            blocked_step: null,
          };
        },
      ),
    );
    render(<SettingsMemoryPage />);

    expect(await screen.findByText("Needs your decision")).toBeDefined();
    expect(screen.getByText(/Added a duplicate column check/)).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Keep current" }));

    await waitFor(() => expect(decide).toHaveBeenCalledTimes(1));
    expect(decide.mock.calls[0][0]).toMatchObject({
      params: { name: "csv-import-checks", versionId: "ver-9" },
      body: { action: "keep_current" },
    });
  });

  it("opens a personal skill's exact version from history and restores it without an expert id", async () => {
    const detailRequests: string[] = [];
    const restore = vi.fn();
    const personal: SkillLearningDetail = {
      skill_name: "csv-import-checks",
      expert_id: null,
      state: "ready",
      state_label: "Ready to use",
      current_version: {
        ...proposal,
        id: "ver-3",
        version: 3,
        expert_id: null,
        state: "ready",
        state_label: "Ready to use",
      },
      open_decision: null,
      versions: [
        {
          ...proposal,
          id: "ver-3",
          version: 3,
          expert_id: null,
          state: "ready",
          state_label: "Ready to use",
        },
        {
          ...proposal,
          id: "ver-2",
          version: 2,
          expert_id: null,
          state: "ready",
          state_label: "Ready to use",
          origin: "saved_during_work",
          origin_label: "Saved during work",
        },
      ],
      policy: { auto_improve: true, learning_paused: false, use_paused: false },
      use: proposal.use!,
    };
    server.use(
      getListSkillLearningHistoryMockHandler200({
        expert_id: null,
        items: [
          {
            id: "ver-3",
            kind: "version",
            skill_name: "csv-import-checks",
            expert_id: null,
            created_at: new Date("2026-09-13T03:00:00Z"),
            summary:
              "Added an encoding check after the previous import failed.",
            origin: "saved_overnight",
            origin_label: "Saved overnight",
            state: "ready",
            state_label: "Ready to use",
            version: 3,
            version_id: "ver-3",
            source: null,
          },
        ],
      }),
      getGetSkillLearningDetailMockHandler200(({ request }) => {
        detailRequests.push(request.url);
        return personal;
      }),
      getRestoreLearnedSkillVersionMockHandler200(async ({ request }) => {
        restore({ url: request.url, body: await request.json() });
        return {
          status: "applied",
          status_label: "Applied",
          reason: "Restored by the owner (from v2)",
          version: null,
          pattern_class: null,
          blocked_step: null,
        };
      }),
    );
    render(<SettingsMemoryPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "csv-import-checks v3" }),
    );
    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    expect(await within(panel).findByTestId("reuse-label")).toBeDefined();
    expect(detailRequests[0]).not.toContain("expert_id");

    await userEvent.click(within(panel).getByRole("tab", { name: "History" }));
    await userEvent.click(
      await within(panel).findByRole("button", { name: "Restore v2" }),
    );
    await waitFor(() => expect(restore).toHaveBeenCalledTimes(1));
    expect(restore.mock.calls[0][0].url).not.toContain("expert_id");
    expect(restore.mock.calls[0][0].body).toEqual({ version_id: "ver-2" });
  });

  it("deep links from a chat chip open the requested personal version", async () => {
    window.history.replaceState(
      {},
      "",
      "/settings/memory?skill=csv-import-checks&version=ver-2",
    );
    server.use(
      getGetSkillLearningDetailMockHandler200({
        skill_name: "csv-import-checks",
        expert_id: null,
        state: "ready",
        state_label: "Ready to use",
        current_version: {
          ...proposal,
          id: "ver-3",
          version: 3,
          expert_id: null,
          state: "ready",
          state_label: "Ready to use",
        },
        open_decision: null,
        versions: [
          {
            ...proposal,
            id: "ver-3",
            version: 3,
            expert_id: null,
            state: "ready",
            state_label: "Ready to use",
          },
          {
            ...proposal,
            id: "ver-2",
            version: 2,
            expert_id: null,
            state: "invalidated",
            state_label: "Archived (source unavailable)",
          },
        ],
        policy: {
          auto_improve: true,
          learning_paused: false,
          use_paused: false,
        },
        use: proposal.use!,
      }),
    );
    render(<SettingsMemoryPage />);

    const panel = await screen.findByRole("complementary", {
      name: "csv-import-checks",
    });
    expect(
      (await within(panel).findByTestId("version-state")).textContent,
    ).toBe("Archived (source unavailable)");
    expect(within(panel).getByText(/Skill now: Ready to use/)).toBeDefined();
    const restore = within(panel).getByRole("button", { name: "Restore v2" });
    expect(restore.hasAttribute("disabled")).toBe(true);
  });
});
