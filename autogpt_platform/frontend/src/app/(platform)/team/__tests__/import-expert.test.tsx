import {
  getGetV1ListProvidersMockHandler,
  getGetV1ListSystemProvidersMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import {
  getListExpertCredentialsMockHandler,
  getListExpertPodsMockHandler,
  getListExpertSetupItemsMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2ListLibraryAgentsMockHandler200,
  getGetV2ListLibraryAgentsResponseMock200,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertPackagePreview } from "@/app/api/__generated__/models/expertPackagePreview";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

const { portabilityFlag, pushMock } = vi.hoisted(() => ({
  portabilityFlag: { enabled: true, ready: true },
  pushMock: vi.fn(),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) => {
      if (flag === "hire-experts") return { enabled: true, ready: true };
      if (flag === "expert-portability") return portabilityFlag;
      return actual.useFlagStatus(flag as never);
    },
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/team",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

const preview: ExpertPackagePreview = {
  manifest: { identity: { name: "Maria", role: "Marketing Strategist" } },
  avatar_kind: "none",
  skills: [
    {
      slug: "brand-voice",
      name: "Brand voice",
      description: "On-brand drafts.",
      files: [{ path: "SKILL.md", size_bytes: 12 }],
    },
  ],
  workflows: [{ index: 0, name: "Calendar", source: "store" }],
};

const imported: Expert = {
  id: "expert-new",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: "",
  skills: ["brand-voice"],
  tagline: null,
  identity: "You are Maria.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
};

function packageFile(name = "maria.expert.zip", sizeBytes?: number) {
  const file = new File([new Uint8Array([0x50, 0x4b])], name, {
    type: "application/zip",
  });
  if (sizeBytes !== undefined) {
    Object.defineProperty(file, "size", { value: sizeBytes });
  }
  return file;
}

/** The header's button and the empty state's link are the same component, so
 *  the tests drive the header one and wait for the roster to settle first. */
async function pickFile(file: File) {
  await screen.findByText("No hired experts yet");
  fireEvent.change(screen.getByTestId("expert-import-input-button"), {
    target: { files: [file] },
  });
}

function parseHandler(response: () => Response) {
  return http.post("/api/proxy/api/experts/import/parse", () => response());
}

function renderTeam() {
  return render(
    <>
      <TeamPage />
      <Toaster />
    </>,
  );
}

beforeEach(() => {
  portabilityFlag.enabled = true;
  pushMock.mockReset();
  server.use(
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertsMockHandler([]),
    getListExpertPodsMockHandler([]),
    getListExpertCredentialsMockHandler([]),
    getListExpertSetupItemsMockHandler([]),
    getGetV1ListProvidersMockHandler([]),
    getGetV1ListSystemProvidersMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200(
      getGetV2ListLibraryAgentsResponseMock200(),
    ),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Importing an expert from a file", () => {
  test("offers the import from the header and the empty state", async () => {
    renderTeam();

    expect(await screen.findByTestId("expert-import-link")).toHaveProperty(
      "textContent",
      "Import from a file",
    );
    expect(screen.getByTestId("expert-import-button")).toHaveProperty(
      "textContent",
      "Import expert",
    );
  });

  test("hides the import while expert portability is off", async () => {
    portabilityFlag.enabled = false;

    renderTeam();

    expect(await screen.findByText("No hired experts yet")).toBeDefined();
    expect(screen.queryByTestId("expert-import-button")).toBeNull();
    expect(screen.queryByTestId("expert-import-link")).toBeNull();
  });

  test("refuses a file that is not a zip, without asking the server", async () => {
    const parsed = vi.fn();
    server.use(
      parseHandler(() => {
        parsed();
        return HttpResponse.json(preview);
      }),
    );

    renderTeam();
    await pickFile(packageFile("notes.txt"));

    expect(await screen.findByText("Can't import this file")).toBeDefined();
    expect(parsed).not.toHaveBeenCalled();
  });

  test("refuses a package over the 20 MiB the route takes", async () => {
    renderTeam();
    await pickFile(packageFile("maria.expert.zip", 21 * 1024 * 1024));

    expect(await screen.findByText("Can't import this file")).toBeDefined();
    expect(screen.getByText(/Maximum size is 20MB/)).toBeDefined();
  });

  test("says so when the server cannot read the package", async () => {
    server.use(
      parseHandler(() =>
        HttpResponse.json({ detail: "Not an expert package" }, { status: 400 }),
      ),
    );

    renderTeam();
    await pickFile(packageFile());

    expect(await screen.findByText("Can't import this file")).toBeDefined();
    expect(screen.getByText("Not an expert package")).toBeDefined();
  });

  test("reviews the package, then imports it with the dialog's edits", async () => {
    const bodies: FormData[] = [];
    server.use(
      parseHandler(() => HttpResponse.json(preview)),
      http.post("/api/proxy/api/experts/import", async ({ request }) => {
        bodies.push(await request.formData());
        return HttpResponse.json(
          { expert: imported, failed_workflows: [], failed_skills: [] },
          { status: 201 },
        );
      }),
    );

    renderTeam();
    await pickFile(packageFile());

    expect(await screen.findByText("Review before importing")).toBeDefined();
    await userEvent.click(
      screen.getByRole("button", { name: "Remove Brand voice" }),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Import expert" }),
    );

    await waitFor(() => expect(bodies).toHaveLength(1));
    const body = bodies[0];
    expect((body.get("file") as File).name).toBe("maria.expert.zip");
    expect(JSON.parse(String(body.get("edits")))).toEqual({
      name: "Maria",
      removed_skill_slugs: ["brand-voice"],
      removed_workflow_indices: [],
      workflows: [],
    });

    expect(await screen.findByText("Imported Maria")).toBeDefined();
    await waitFor(() =>
      expect(pushMock).toHaveBeenCalledWith("/team/expert-new"),
    );
  });

  test("names what did not make it into a partial import", async () => {
    server.use(
      parseHandler(() => HttpResponse.json(preview)),
      http.post("/api/proxy/api/experts/import", () =>
        HttpResponse.json(
          {
            expert: imported,
            failed_workflows: ["Calendar"],
            failed_skills: ["brand-voice"],
          },
          { status: 201 },
        ),
      ),
    );

    renderTeam();
    await pickFile(packageFile());

    await screen.findByText("Review before importing");
    await userEvent.click(
      screen.getByRole("button", { name: "Import expert" }),
    );

    expect(
      await screen.findByText(
        "1 workflow and 1 skill couldn't be imported: Calendar, brand-voice",
      ),
    ).toBeDefined();
  });

  test("keeps the dialog open and says so when the import fails", async () => {
    server.use(
      parseHandler(() => HttpResponse.json(preview)),
      http.post("/api/proxy/api/experts/import", () =>
        HttpResponse.json({ detail: "Your team is full" }, { status: 409 }),
      ),
    );

    renderTeam();
    await pickFile(packageFile());

    await screen.findByText("Review before importing");
    await userEvent.click(
      screen.getByRole("button", { name: "Import expert" }),
    );

    expect(
      await screen.findByText("Couldn't import this expert"),
    ).toBeDefined();
    expect(screen.getByText("Your team is full")).toBeDefined();
    expect(screen.getByText("Review before importing")).toBeDefined();
    expect(pushMock).not.toHaveBeenCalled();
  });
});
