import {
  getGetExpertActivityMockHandler,
  getGetExpertMockHandler,
  getListExpertRunsMockHandler,
  getUpdateExpertModeMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import {
  getGetHomeDashboardMockHandler,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

vi.mock("@/services/environment", async (importActual) => {
  const actual = await importActual<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: {
      ...actual.environment,
      getAGPTServerBaseUrl: () => "http://localhost:18006",
    },
  };
});

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

const flags = vi.hoisted(() => ({ autoMode: true }));
const toastMock = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: toastMock };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: () => ({ enabled: true, ready: true }),
    useGetFlag: (flag: string) =>
      flag === actual.Flag.COPILOT_AUTO_MODE ? flags.autoMode : false,
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
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
  bio: null,
  skills: [],
  tagline: null,
  identity: "You are Maria.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

let expertResponse: Expert;
let patchBodies: Record<string, unknown>[];

beforeEach(() => {
  flags.autoMode = true;
  toastMock.mockReset();
  expertResponse = maria;
  patchBodies = [];
  server.use(
    getGetHomeDashboardMockHandler(
      getGetHomeDashboardResponseMock200({ attention: [] }),
    ),
    getGetExpertMockHandler(() => expertResponse),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertRunsMockHandler([]),
    getGetExpertActivityMockHandler({ timezone: "UTC", days: [] }),
    getUpdateExpertModeMockHandler(async ({ request }) => {
      const body = (await request.clone().json()) as Record<string, unknown>;
      patchBodies.push(body);
      expertResponse = {
        ...maria,
        autopilot_mode: body.autopilot_mode as Expert["autopilot_mode"],
      };
      return expertResponse;
    }),
  );
});

afterEach(() => {
  window.localStorage.removeItem("team-workflows-view");
});

async function openSettings() {
  const user = userEvent.setup();
  await user.click(await screen.findByRole("tab", { name: "Settings" }));
  return user;
}

describe("Expert approval mode", () => {
  test("starts on Auto and saves a picked mode for the expert", async () => {
    render(<ExpertDetailPage />);
    const user = await openSettings();

    const section = await screen.findByRole("region", {
      name: "Maria approval mode",
    });
    expect(section.textContent).toContain("Routines, scheduled runs");
    expect(
      screen.getByRole("radio", { name: /auto \(default\)/i }),
    ).toHaveProperty("checked", true);

    await user.click(screen.getByRole("radio", { name: /ask first/i }));

    await waitFor(() =>
      expect(patchBodies).toEqual([{ autopilot_mode: "ask_first" }]),
    );
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: /ask first/i })).toHaveProperty(
        "checked",
        true,
      ),
    );
    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith({
        title: "Approval mode updated",
      }),
    );
  });

  test("shows the stored default on load", async () => {
    expertResponse = { ...maria, autopilot_mode: "unsupervised" };
    render(<ExpertDetailPage />);
    await openSettings();

    expect(
      await screen.findByRole("radio", { name: /unsupervised/i }),
    ).toHaveProperty("checked", true);
  });

  test("saves Unsupervised only after the confirm", async () => {
    render(<ExpertDetailPage />);
    const user = await openSettings();

    await user.click(
      await screen.findByRole("radio", { name: /unsupervised/i }),
    );
    expect(
      await screen.findByRole("heading", {
        name: "Run Maria unsupervised by default?",
      }),
    ).toBeDefined();
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("heading", {
          name: "Run Maria unsupervised by default?",
        }),
      ).toBeNull(),
    );
    expect(patchBodies).toEqual([]);
    expect(
      screen.getByRole("radio", { name: /auto \(default\)/i }),
    ).toHaveProperty("checked", true);

    await user.click(screen.getByRole("radio", { name: /unsupervised/i }));
    await user.click(
      await screen.findByRole("button", { name: "Run unsupervised" }),
    );

    await waitFor(() =>
      expect(patchBodies).toEqual([{ autopilot_mode: "unsupervised" }]),
    );
  });

  test("explains when approval modes are off for the account", async () => {
    server.use(
      http.patch("*/api/experts/expert-maria/mode", () =>
        HttpResponse.json({ detail: "feature_disabled" }, { status: 403 }),
      ),
    );
    render(<ExpertDetailPage />);
    const user = await openSettings();

    await user.click(await screen.findByRole("radio", { name: /ask first/i }));

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Approval modes are not enabled for your account",
          variant: "destructive",
        }),
      ),
    );
  });

  test("is hidden while the flag is off", async () => {
    flags.autoMode = false;
    render(<ExpertDetailPage />);
    await openSettings();

    expect(await screen.findByText("Danger zone")).toBeDefined();
    expect(
      screen.queryByRole("region", { name: "Maria approval mode" }),
    ).toBeNull();
  });
});
