import {
  getDeleteCopilotSkillMockHandler,
  getDeleteCopilotSkillMockHandler422,
  getListCopilotSkillsMockHandler,
} from "@/app/api/__generated__/endpoints/skills/skills.msw";
import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import SkillsPage from "../page";

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    useToast: () => ({ toast: toastMock }),
  };
});

function makeSkill(overrides: Partial<CopilotSkillInfo>): CopilotSkillInfo {
  return {
    name: "oauth_flow",
    description: "OAuth handshake recipe for Google and GitHub providers.",
    triggers: ["connect_integration"],
    ...overrides,
  };
}

describe("SkillsPage", () => {
  beforeEach(() => {
    toastMock.mockClear();
  });

  afterEach(() => {
    server.resetHandlers();
  });

  test("renders empty state when no skills exist", async () => {
    server.use(getListCopilotSkillsMockHandler([]));

    render(<SkillsPage />);

    expect(await screen.findByTestId("skills-empty")).toBeDefined();
    expect(screen.queryByTestId("skills-list")).toBeNull();
  });

  test("renders one row per user skill returned by the API", async () => {
    server.use(
      getListCopilotSkillsMockHandler([
        makeSkill({ name: "oauth_flow", description: "OAuth handshake" }),
        makeSkill({
          name: "cleanup_workspace",
          description: "Workspace cleanup",
          triggers: [],
        }),
      ]),
    );

    render(<SkillsPage />);

    const rows = await screen.findAllByTestId("skill-row");
    expect(rows).toHaveLength(2);
    expect(rows[0].getAttribute("data-skill-name")).toBe("oauth_flow");
    expect(rows[1].getAttribute("data-skill-name")).toBe("cleanup_workspace");
    expect(screen.getByText("OAuth handshake")).toBeDefined();
    expect(screen.getByText("Workspace cleanup")).toBeDefined();
  });

  test("renders trigger chips when the skill has triggers", async () => {
    server.use(
      getListCopilotSkillsMockHandler([
        makeSkill({
          name: "oauth_flow",
          triggers: ["connect_integration", "refresh_token"],
        }),
      ]),
    );

    render(<SkillsPage />);

    const row = await screen.findByTestId("skill-row");
    const triggers = within(row).getByTestId("skill-triggers");
    expect(within(triggers).getByText("connect_integration")).toBeDefined();
    expect(within(triggers).getByText("refresh_token")).toBeDefined();
  });

  test("Delete button opens the confirmation dialog and calls the delete API", async () => {
    server.use(
      getListCopilotSkillsMockHandler([makeSkill({ name: "oauth_flow" })]),
      getDeleteCopilotSkillMockHandler(),
    );

    render(<SkillsPage />);

    const deleteButton = await screen.findByTestId("skill-delete-button");
    fireEvent.click(deleteButton);

    const confirmButton = await screen.findByTestId("skill-confirm-delete");
    fireEvent.click(confirmButton);

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Skill deleted" }),
      );
    });
  });

  test("shows a destructive toast when the delete API fails", async () => {
    server.use(
      getListCopilotSkillsMockHandler([makeSkill({ name: "oauth_flow" })]),
      getDeleteCopilotSkillMockHandler422(),
    );

    render(<SkillsPage />);

    const deleteButton = await screen.findByTestId("skill-delete-button");
    fireEvent.click(deleteButton);

    const confirmButton = await screen.findByTestId("skill-confirm-delete");
    fireEvent.click(confirmButton);

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to delete skill",
          variant: "destructive",
        }),
      );
    });
  });
});
