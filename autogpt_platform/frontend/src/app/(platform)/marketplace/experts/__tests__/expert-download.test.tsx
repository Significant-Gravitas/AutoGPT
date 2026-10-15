import {
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListSystemProvidersMockHandler } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ExpertPage as MarketplaceExpertPage } from "../[expertId]/components/ExpertPage";

vi.mock("posthog-js", () => ({ default: { capture: vi.fn() } }));

const mockUseAuth = vi.hoisted(() => vi.fn());
const flagStatusMock = vi.hoisted(() =>
  vi.fn(() => ({ enabled: true, ready: true })),
);

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace/experts/template-maria",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ expertId: "template-maria" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? flagStatusMock()
        : actual.useFlagStatus(flag as never),
  };
});

const PACKAGE_URL = "/api/proxy/api/experts/templates/:templateId/package";
const ZIP_BYTES = new Uint8Array([0x50, 0x4b, 0x03, 0x04]);

const mariaTemplate: Expert = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: "Maria is a senior marketing strategist.",
  tagline: "Grows your brand while you sleep",
  skills: [],
  identity: "You are Maria.",
  voice_preferences: "Warm.",
  boundaries: "Never invent evidence.",
  protected_soul_rules: [],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  weekly_budget: 500,
  workflows: [],
};

const hiredMaria: Expert = {
  ...mariaTemplate,
  id: "expert-maria",
  is_template: false,
  source_template_id: "template-maria",
};

function zipHandler(requested: string[] = []) {
  return http.get(PACKAGE_URL, ({ params }) => {
    requested.push(String(params.templateId));
    return new HttpResponse(new Blob([ZIP_BYTES]), {
      headers: {
        "Content-Type": "application/zip",
        "Content-Disposition": 'attachment; filename="maria.expert.zip"',
      },
    });
  });
}

const createObjectURL = vi.fn((_: Blob | MediaSource) => "blob:mock-url");
const originalCreate = URL.createObjectURL;
const originalRevoke = URL.revokeObjectURL;
let saved: string[] = [];

function renderPage() {
  return render(
    <>
      <MarketplaceExpertPage />
      <Toaster />
    </>,
  );
}

describe("Marketplace expert page download", () => {
  beforeEach(() => {
    saved = [];
    createObjectURL.mockClear();
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
      this: HTMLAnchorElement,
    ) {
      saved.push(this.download);
    });
    flagStatusMock.mockReturnValue({ enabled: true, ready: true });
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getGetV1ListSystemProvidersMockHandler([]),
      getListExpertTemplatesMockHandler([mariaTemplate]),
      getListExpertsMockHandler([]),
    );
  });

  afterEach(() => {
    URL.createObjectURL = originalCreate;
    URL.revokeObjectURL = originalRevoke;
    vi.restoreAllMocks();
  });

  test("saves the template as a .expert.zip from beside the hire button", async () => {
    const requested: string[] = [];
    server.use(zipHandler(requested));

    renderPage();

    const hire = await screen.findByRole("button", { name: "Hire Maria" });
    const download = screen.getByRole("button", { name: "Download as file" });
    expect(hire.parentElement).toBe(download.parentElement);

    await userEvent.click(download);

    await vi.waitFor(() => expect(saved).toEqual(["maria.expert.zip"]));
    expect(requested).toEqual(["template-maria"]);
    const blob = createObjectURL.mock.calls[0][0] as Blob;
    expect(blob).toBeInstanceOf(Blob);
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(ZIP_BYTES);
  });

  test("can be reached with Tab and triggered with Enter", async () => {
    server.use(zipHandler());

    renderPage();

    const download = await screen.findByRole("button", {
      name: "Download as file",
    });
    for (let i = 0; i < 50 && document.activeElement !== download; i++) {
      await userEvent.tab();
    }
    expect(document.activeElement).toBe(download);
    await userEvent.keyboard("{Enter}");

    await vi.waitFor(() => expect(saved).toEqual(["maria.expert.zip"]));
  });

  test("is offered beside 'On your team' once the expert is hired", async () => {
    server.use(getListExpertsMockHandler([hiredMaria]), zipHandler());

    renderPage();

    const badge = await screen.findByText("On your team");
    const row = badge.closest("div") as HTMLElement;
    await userEvent.click(
      within(row).getByRole("button", { name: "Download as file" }),
    );

    await vi.waitFor(() => expect(saved).toEqual(["maria.expert.zip"]));
  });

  test("is not offered to signed-out visitors", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });

    renderPage();

    expect(
      await screen.findByRole("link", { name: "Get started" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Download as file" }),
    ).toBeNull();
  });

  test("is not offered while hiring is closed", async () => {
    flagStatusMock.mockReturnValue({ enabled: false, ready: true });

    renderPage();

    expect(
      await screen.findByRole("heading", { level: 1, name: "Maria" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Download as file" }),
    ).toBeNull();
  });

  test("shows a toast and saves nothing when the download fails", async () => {
    server.use(
      http.get(PACKAGE_URL, () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    renderPage();

    await userEvent.click(
      await screen.findByRole("button", { name: "Download as file" }),
    );

    expect(
      await screen.findByText("Couldn't download this expert"),
    ).toBeDefined();
    expect(
      screen.getByText("Something went wrong. Try again in a moment."),
    ).toBeDefined();
    expect(saved).toEqual([]);
  });

  test("says when the expert is too large to download", async () => {
    server.use(
      http.get(PACKAGE_URL, () =>
        HttpResponse.json(
          { detail: "Package is over the size limit" },
          { status: 413 },
        ),
      ),
    );

    renderPage();

    await userEvent.click(
      await screen.findByRole("button", { name: "Download as file" }),
    );

    expect(
      await screen.findByText(
        "This expert is too large to download as a file.",
      ),
    ).toBeDefined();
    expect(saved).toEqual([]);
  });
});
