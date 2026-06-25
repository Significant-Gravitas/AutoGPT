import {
  getDeleteCopilotSkillMockHandler,
  getDeleteCopilotSkillMockHandler422,
  getListCopilotSkillsMockHandler,
  getReadCopilotSkillMockHandler,
  getReadCopilotSkillMockHandler404,
  getUploadCopilotSkillMockHandler201,
  getUploadCopilotSkillMockHandler409,
} from "@/app/api/__generated__/endpoints/skills/skills.msw";
import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import type { CopilotSkillDetail } from "@/app/api/__generated__/models/copilotSkillDetail";
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

  test("View dialog opens, fetches the detail, and renders the body", async () => {
    const detail: CopilotSkillDetail = {
      name: "oauth_flow",
      description: "OAuth handshake recipe for Google and GitHub providers.",
      triggers: ["connect_integration"],
      body: "## Why\n\nProviders share an OAuth dance.\n\n## Steps\n1. Hit /authorize\n2. Trade code for token",
      version: null,
      is_default: false,
      sibling_files: [],
    };
    server.use(
      getListCopilotSkillsMockHandler([makeSkill({ name: "oauth_flow" })]),
      getReadCopilotSkillMockHandler(detail),
    );

    render(<SkillsPage />);

    const viewButton = await screen.findByTestId("skill-view-button");
    fireEvent.click(viewButton);

    // Body is rendered (not the loading spinner / error card).
    const body = await screen.findByTestId("skill-view-body");
    expect(body.textContent).toContain("Hit /authorize");
    // No sibling-files section when sibling_files is empty.
    expect(screen.queryByTestId("skill-view-sibling-files")).toBeNull();
    // No error state when fetch succeeds.
    expect(screen.queryByTestId("skill-view-error")).toBeNull();
  });

  test("View dialog lists sibling files when the skill bundle has them", async () => {
    const detail: CopilotSkillDetail = {
      name: "oauth_flow",
      description: "OAuth handshake recipe.",
      triggers: [],
      body: "# Body",
      version: null,
      is_default: false,
      sibling_files: [
        "/skills/oauth_flow/references/providers.md",
        "/skills/oauth_flow/scripts/exchange_code.py",
      ],
    };
    server.use(
      getListCopilotSkillsMockHandler([makeSkill({ name: "oauth_flow" })]),
      getReadCopilotSkillMockHandler(detail),
    );

    render(<SkillsPage />);

    const viewButton = await screen.findByTestId("skill-view-button");
    fireEvent.click(viewButton);

    const siblings = await screen.findByTestId("skill-view-sibling-files");
    expect(
      within(siblings).getByText("/skills/oauth_flow/references/providers.md"),
    ).toBeDefined();
    expect(
      within(siblings).getByText("/skills/oauth_flow/scripts/exchange_code.py"),
    ).toBeDefined();
  });

  test("View dialog shows an error card when the detail fetch fails", async () => {
    server.use(
      getListCopilotSkillsMockHandler([makeSkill({ name: "oauth_flow" })]),
      getReadCopilotSkillMockHandler404(),
    );

    render(<SkillsPage />);

    const viewButton = await screen.findByTestId("skill-view-button");
    fireEvent.click(viewButton);

    // Error path: ErrorCard wrapper is rendered, body pre is not.
    expect(await screen.findByTestId("skill-view-error")).toBeDefined();
    expect(screen.queryByTestId("skill-view-body")).toBeNull();
  });

  test("Download button fetches the skill detail and triggers a file download", async () => {
    // jsdom doesn't implement these blob helpers — patch just the two
    // methods (not the whole URL constructor, which the fetch mutator needs).
    const createObjectURL = vi.fn(() => "blob:mock-url");
    const revokeObjectURL = vi.fn();
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = revokeObjectURL;
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    try {
      const detail: CopilotSkillDetail = {
        name: "oauth_flow",
        description: "OAuth handshake recipe.",
        triggers: ["connect_integration"],
        body: "## Steps\n1. Hit /authorize",
        version: null,
        is_default: false,
        sibling_files: [],
      };
      server.use(
        getListCopilotSkillsMockHandler([makeSkill({ name: "oauth_flow" })]),
        getReadCopilotSkillMockHandler(detail),
      );

      render(<SkillsPage />);

      const downloadButton = await screen.findByTestId("skill-download-button");
      fireEvent.click(downloadButton);

      await vi.waitFor(() => {
        expect(createObjectURL).toHaveBeenCalled();
        expect(clickSpy).toHaveBeenCalled();
      });
    } finally {
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
      clickSpy.mockRestore();
    }
  });

  test("Upload button sends the picked file and refreshes the list", async () => {
    server.use(
      getListCopilotSkillsMockHandler([]),
      getUploadCopilotSkillMockHandler201({
        name: "uploaded_skill",
        description: "An uploaded recipe.",
        triggers: [],
      }),
    );

    render(<SkillsPage />);

    await screen.findByTestId("skills-empty");

    const input = screen.getByTestId("skill-upload-input");
    const file = new File(
      [
        "---\nname: uploaded_skill\ndescription: An uploaded recipe.\n---\n\n# Body\n",
      ],
      "uploaded_skill.md",
      { type: "text/markdown" },
    );
    fireEvent.change(input, { target: { files: [file] } });

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: expect.stringContaining("uploaded"),
        }),
      );
    });
  });

  test("Upload rejects an over-long description client-side with the exact length", async () => {
    // No upload handler registered — if the code POSTed, it would not match
    // and the test would surface a different failure. The client-side
    // pre-flight should short-circuit before any request.
    server.use(getListCopilotSkillsMockHandler([]));

    render(<SkillsPage />);
    await screen.findByTestId("skills-empty");

    const input = screen.getByTestId("skill-upload-input");
    const longDescription = "x".repeat(201);
    const file = new File(
      [`---\nname: too_long\ndescription: ${longDescription}\n---\n\nbody`],
      "too_long.md",
      { type: "text/markdown" },
    );
    fireEvent.change(input, { target: { files: [file] } });

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Can't upload this skill",
          description: expect.stringContaining("201/200"),
          variant: "destructive",
        }),
      );
    });
  });

  test("Upload shows a destructive toast when the skill limit is reached", async () => {
    server.use(
      getListCopilotSkillsMockHandler([]),
      getUploadCopilotSkillMockHandler409(),
    );

    render(<SkillsPage />);

    await screen.findByTestId("skills-empty");

    const input = screen.getByTestId("skill-upload-input");
    const file = new File(
      ["---\nname: x\ndescription: y\n---\n\nbody"],
      "x.md",
      {
        type: "text/markdown",
      },
    );
    fireEvent.change(input, { target: { files: [file] } });

    await vi.waitFor(() => {
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Failed to upload skill",
          variant: "destructive",
        }),
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
