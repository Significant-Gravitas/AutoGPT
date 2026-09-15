import {
  getListExpertsMockHandler,
  getListExpertTemplatesMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListSystemProvidersMockHandler } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ExpertPage } from "../[expertId]/components/ExpertPage";

const mockUseAuth = vi.hoisted(() => vi.fn());
const { portabilityFlag } = vi.hoisted(() => ({
  portabilityFlag: { enabled: true, ready: true },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/marketplace/experts/template-maria",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ expertId: "template-maria" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

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

const mariaTemplate: Expert = {
  id: "template-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: "Maria is a senior marketing strategist.",
  skills: ["Content strategy"],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  voice_preferences: "Warm, concise, and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [],
  is_template: true,
  source_template_id: null,
  is_archived: false,
  workflows: [],
};

const DOWNLOAD_BUTTON = { name: "Download as file" };

beforeEach(() => {
  portabilityFlag.enabled = true;
  mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
  server.use(
    getGetV1ListSystemProvidersMockHandler([]),
    getListExpertTemplatesMockHandler([mariaTemplate]),
    getListExpertsMockHandler([]),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Downloading a marketplace expert", () => {
  test("downloads the template package, not the viewer's own expert", async () => {
    const requested: string[] = [];
    server.use(
      http.get(
        "/api/proxy/api/experts/templates/:templateId/package",
        ({ request }) => {
          requested.push(new URL(request.url).pathname);
          return new HttpResponse(new Uint8Array([0x50, 0x4b, 0x03, 0x04]), {
            headers: { "Content-Type": "application/zip" },
          });
        },
      ),
    );
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    render(<ExpertPage />);
    await userEvent.click(await screen.findByRole("button", DOWNLOAD_BUTTON));

    await waitFor(() => expect(click).toHaveBeenCalledTimes(1));
    expect(requested).toEqual([
      "/api/proxy/api/experts/templates/template-maria/package",
    ]);
  });

  test("hides the button from signed-out visitors", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });

    render(<ExpertPage />);

    expect(
      await screen.findByRole("link", { name: "Get started" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", DOWNLOAD_BUTTON)).toBeNull();
  });

  test("hides the button while expert portability is off", async () => {
    portabilityFlag.enabled = false;

    render(<ExpertPage />);

    expect(
      await screen.findByRole("button", { name: "Hire Maria" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", DOWNLOAD_BUTTON)).toBeNull();
  });
});
