import {
  getGetExpertActivityMockHandler,
  getGetExpertMockHandler,
  getListExpertRunsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

const { portabilityFlag } = vi.hoisted(() => ({
  portabilityFlag: { enabled: true, ready: true },
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
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/team/expert-maria",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ expertId: "expert-maria" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const maria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: "Maria is a senior marketing strategist.",
  skills: ["Content strategy", "Positioning"],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

const ZIP_BYTES = new Uint8Array([0x50, 0x4b, 0x03, 0x04, 0x00]);
const EXPORT_BUTTON = { name: "Download as file" };

function packageHandler(contentDisposition?: string) {
  const headers: Record<string, string> = { "Content-Type": "application/zip" };
  if (contentDisposition) headers["Content-Disposition"] = contentDisposition;

  return http.get("/api/proxy/api/experts/:expertId/package", () => {
    return new HttpResponse(ZIP_BYTES, { headers });
  });
}

/** `downloadFile` builds an anchor, sets `download`, clicks it and throws it
 *  away, so the name it picked is only readable from the click itself. */
function spyOnDownload() {
  const names: string[] = [];
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  const click = vi
    .spyOn(HTMLAnchorElement.prototype, "click")
    .mockImplementation(function (this: HTMLAnchorElement) {
      names.push(this.download);
    });

  return { names, click };
}

function renderPage() {
  return render(
    <>
      <ExpertDetailPage />
      <Toaster />
    </>,
  );
}

beforeEach(() => {
  portabilityFlag.enabled = true;
  server.use(
    getGetExpertMockHandler(maria),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertRunsMockHandler([]),
    getGetExpertActivityMockHandler({ timezone: "UTC", days: [] }),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Exporting an expert from its page", () => {
  test("downloads the package as the file the server named", async () => {
    server.use(packageHandler('attachment; filename="maria-ops.expert.zip"'));
    const { names, click } = spyOnDownload();

    renderPage();
    await userEvent.click(await screen.findByRole("button", EXPORT_BUTTON));

    await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    expect(names).toEqual(["maria-ops.expert.zip"]);
  });

  test("names the file after the expert when the response does not", async () => {
    server.use(packageHandler());
    const { names, click } = spyOnDownload();

    renderPage();
    await userEvent.click(await screen.findByRole("button", EXPORT_BUTTON));

    await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    expect(names).toEqual(["maria.expert.zip"]);
  });

  test("keeps the expert on the page with a toast when the export fails", async () => {
    server.use(
      http.get("/api/proxy/api/experts/:expertId/package", () =>
        HttpResponse.json({ detail: "Could not package it" }, { status: 500 }),
      ),
    );
    const { click } = spyOnDownload();

    renderPage();
    await userEvent.click(await screen.findByRole("button", EXPORT_BUTTON));

    expect(
      await screen.findByText("Couldn't download this expert"),
    ).toBeDefined();
    expect(await screen.findByText("Could not package it")).toBeDefined();
    expect(click).not.toHaveBeenCalled();
  });

  test("hides the button while expert portability is off", async () => {
    portabilityFlag.enabled = false;

    renderPage();

    expect(await screen.findByRole("heading", { name: "Maria" })).toBeDefined();
    expect(screen.queryByRole("button", EXPORT_BUTTON)).toBeNull();
  });
});
