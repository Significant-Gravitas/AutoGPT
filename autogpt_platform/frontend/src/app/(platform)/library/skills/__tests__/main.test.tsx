import {
  getDeleteCopilotSkillMockHandler,
  getDeleteCopilotSkillMockHandler422,
  getListCopilotSkillsMockHandler,
  getReadCopilotSkillMockHandler,
  getReadCopilotSkillMockHandler404,
  getUploadCopilotSkillMockHandler201,
  getUploadCopilotSkillMockHandler409,
  getUploadCopilotSkillPackageMockHandler201,
} from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { http, HttpResponse } from "msw";
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

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: pushMock,
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/library/skills",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

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

function makePackageDetail(): CopilotSkillDetail {
  return {
    name: "oauth_flow",
    description: "OAuth handshake recipe.",
    triggers: [],
    body: "# Body",
    version: null,
    is_default: false,
    files: [
      {
        path: "references/providers.md",
        size_bytes: 2048,
        is_executable: false,
      },
      { path: "references/errors.md", size_bytes: 300, is_executable: false },
      {
        path: "scripts/exchange_code.py",
        size_bytes: 512,
        is_executable: true,
      },
    ],
  };
}

describe("SkillsPage", () => {
  beforeEach(() => {
    toastMock.mockClear();
    pushMock.mockClear();
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
      files: [],
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
    // No file tree when the skill is a lone SKILL.md.
    expect(screen.queryByTestId("skill-view-files")).toBeNull();
    // No error state when fetch succeeds.
    expect(screen.queryByTestId("skill-view-error")).toBeNull();
  });

  test("View dialog renders the package as a tree of directories and sized files", async () => {
    server.use(
      getListCopilotSkillsMockHandler([makeSkill({ name: "oauth_flow" })]),
      getReadCopilotSkillMockHandler(makePackageDetail()),
    );

    render(<SkillsPage />);

    fireEvent.click(await screen.findByTestId("skill-view-button"));

    const tree = await screen.findByTestId("skill-view-files");
    // Each directory appears once, above its own files, and only files carry
    // a size.
    expect(
      Array.from(tree.querySelectorAll("li")).map((li) => li.textContent),
    ).toEqual([
      "references/",
      "errors.md300 B",
      "providers.md2.0 KB",
      "scripts/",
      "exchange_code.py512 B",
    ]);
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
        files: [],
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

  test("Download gives a .zip when the skill carries package files", async () => {
    const createObjectURL = vi.fn((_: Blob | MediaSource) => "blob:mock-url");
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = vi.fn();
    const saved: string[] = [];
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        saved.push(this.download);
      });

    try {
      server.use(
        getListCopilotSkillsMockHandler([makeSkill({ name: "oauth_flow" })]),
        getReadCopilotSkillMockHandler(makePackageDetail()),
        // The generated handler hardcodes application/json, which would send
        // the archive through the mutator's text path — mock the zip itself.
        http.get(
          "/api/proxy/api/skills/:name/package",
          () =>
            new HttpResponse(
              new Blob([new Uint8Array([0x50, 0x4b, 0x03, 0x04])]),
              {
                headers: { "Content-Type": "application/zip" },
              },
            ),
        ),
      );

      render(<SkillsPage />);

      fireEvent.click(await screen.findByTestId("skill-download-button"));

      await vi.waitFor(() => expect(saved).toEqual(["oauth_flow.zip"]));
      // The archive must survive the fetch layer as bytes; read as text it
      // would download corrupted and open in nothing.
      expect(createObjectURL.mock.calls[0][0]).toBeInstanceOf(Blob);
    } finally {
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
      clickSpy.mockRestore();
    }
  });

  test("a picked .zip goes to the package endpoint and a .md to the single-file one", async () => {
    const posted: string[] = [];
    const record = ({ request }: { request: Request }) => {
      if (request.method === "POST") posted.push(new URL(request.url).pathname);
    };
    server.events.on("request:start", record);

    try {
      server.use(
        getListCopilotSkillsMockHandler([]),
        getUploadCopilotSkillMockHandler201({
          name: "from_md",
          description: "d",
          triggers: [],
        }),
        getUploadCopilotSkillPackageMockHandler201({
          name: "from_zip",
          description: "d",
          triggers: [],
        }),
      );

      render(<SkillsPage />);
      await screen.findByTestId("skills-empty");
      const input = screen.getByTestId("skill-upload-input");
      expect(input.getAttribute("accept")).toContain(".zip");

      fireEvent.change(input, {
        target: {
          files: [
            new File([new Uint8Array([0x50, 0x4b, 0x03, 0x04])], "pkg.zip", {
              type: "application/zip",
            }),
          ],
        },
      });
      await vi.waitFor(() =>
        expect(posted).toEqual(["/api/proxy/api/skills/package"]),
      );

      fireEvent.change(input, {
        target: {
          files: [
            new File(
              ["---\nname: from_md\ndescription: d\n---\n\nbody"],
              "from_md.md",
              { type: "text/markdown" },
            ),
          ],
        },
      });
      await vi.waitFor(() =>
        expect(posted).toEqual([
          "/api/proxy/api/skills/package",
          "/api/proxy/api/skills",
        ]),
      );
    } finally {
      server.events.removeListener("request:start", record);
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
    const longDescription = "x".repeat(1025);
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
          description: expect.stringContaining("1025/1024"),
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

  test("header shows New skill button in empty state and it starts the guided flow", async () => {
    server.use(getListCopilotSkillsMockHandler([]));

    render(<SkillsPage />);
    await screen.findByTestId("skills-empty");

    fireEvent.click(screen.getByTestId("skill-new-button"));

    await vi.waitFor(() => {
      expect(pushMock).toHaveBeenCalledWith(
        expect.stringContaining("/copilot#prompt="),
      );
    });
    const url = pushMock.mock.calls[0][0] as string;
    expect(decodeURIComponent(url.split("#prompt=")[1])).toContain(
      "I want to teach you a new skill",
    );
  });

  test("empty state describes skills for experts", async () => {
    server.use(getListCopilotSkillsMockHandler([]));

    render(<SkillsPage />);

    const empty = await screen.findByTestId("skills-empty");
    expect(empty.textContent).toContain("No skills yet");
    expect(empty.textContent).toContain(
      "Give your experts a repeatable process to follow.",
    );
  });

  test("uploaded skill shows a New badge", async () => {
    server.use(
      getListCopilotSkillsMockHandler([
        makeSkill({
          name: "uploaded_skill",
          description: "An uploaded recipe.",
          triggers: [],
        }),
      ]),
      getUploadCopilotSkillMockHandler201({
        name: "uploaded_skill",
        description: "An uploaded recipe.",
        triggers: [],
      }),
    );

    render(<SkillsPage />);
    await screen.findAllByTestId("skill-row");
    expect(screen.queryByTestId("skill-new-badge")).toBeNull();

    const input = screen.getByTestId("skill-upload-input");
    const file = new File(
      [
        "---\nname: uploaded_skill\ndescription: An uploaded recipe.\n---\n\n# Body\n",
      ],
      "uploaded_skill.md",
      { type: "text/markdown" },
    );
    fireEvent.change(input, { target: { files: [file] } });

    expect(await screen.findByTestId("skill-new-badge")).toBeDefined();
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
