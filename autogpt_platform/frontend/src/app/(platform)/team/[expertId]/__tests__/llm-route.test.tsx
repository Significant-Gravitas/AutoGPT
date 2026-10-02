import { getGetV2ListChatTransportsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getGetExpertMockHandler,
  getListExpertRunsMockHandler,
  getUpdateExpertLlmRouteMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetHomeDashboardMockHandler,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ExpertDetailPage from "../page";

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: () => ({ enabled: true, ready: true }),
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

const maria = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  color: "",
  role: "Marketing Strategist",
  tagline: null,
  bio: "Maria is a senior marketing strategist.",
  skills: [],
  identity: "You are Maria.",
  voice_preferences: "Direct.",
  voice_samples: [],
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
  weekly_budget: 5000,
  weekly_spend: 1200,
  schedules_paused_at: null,
  pod_id: null,
  llm_auth_provider: null,
  llm_credential_id: null,
  llm_route_label: null,
  llm_route_available: true,
} as unknown as Expert;

const mariaOnChatGPT: Expert = {
  ...maria,
  llm_auth_provider: "codex",
  llm_credential_id: "cred-1",
  llm_route_label: "ChatGPT",
  llm_route_available: true,
};

const platform: ChatTransportResponse = {
  auth_provider: "platform",
  credential_id: null,
  label: "AutoGPT Platform",
  available: true,
  default: true,
};

const chatgpt: ChatTransportResponse = {
  auth_provider: "codex",
  credential_id: "cred-1",
  label: "ChatGPT",
  available: true,
  default: false,
};

const microsoft: ChatTransportResponse = {
  auth_provider: "microsoft_365_copilot",
  credential_id: "cred-msft",
  label: "Microsoft 365 Copilot",
  available: true,
  default: false,
};

beforeEach(() => {
  server.use(
    getGetHomeDashboardMockHandler(
      getGetHomeDashboardResponseMock200({ attention: [] }),
    ),
    getGetExpertMockHandler(maria),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertRunsMockHandler([]),
    getGetV2ListChatTransportsMockHandler200({
      transports: [platform, chatgpt, microsoft],
    }),
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json([
        {
          id: "cred-1",
          provider: "codex",
          type: "oauth2",
          title: "ChatGPT",
          username: "maria@example.com",
        },
      ]),
    ),
  );
});

async function openTab(name: string) {
  await userEvent.click(await screen.findByRole("tab", { name }));
}

async function findRouteSection() {
  return screen.findByRole("region", { name: "Maria AI connection" });
}

async function findRouteSelect() {
  const section = await findRouteSection();
  const select = within(section).getByRole("combobox");
  await waitFor(() =>
    expect((select as HTMLButtonElement).disabled).toBe(false),
  );
  return select;
}

describe("an expert's AI connection", () => {
  it("follows the account default until the owner pins one", async () => {
    render(<ExpertDetailPage />);

    await openTab("Settings");
    const select = await findRouteSelect();
    expect(select.textContent).toContain("Account default");
    expect(
      within(await findRouteSection()).getByText(
        "Follows the default AI connection chosen in Settings.",
      ),
    ).toBeDefined();
    expect(
      within(await findRouteSection()).getByRole("link", {
        name: "Open Settings",
      }),
    ).toHaveProperty("href", expect.stringContaining("/settings/integrations"));
  });

  it("pins a connection through the API and names the account it runs as", async () => {
    let patchBody: unknown;
    server.use(
      getUpdateExpertLlmRouteMockHandler(async ({ request }) => {
        patchBody = await request.json();
        server.use(getGetExpertMockHandler(mariaOnChatGPT));
        return mariaOnChatGPT;
      }),
    );
    render(<ExpertDetailPage />);

    await openTab("Settings");
    fireEvent.click(await findRouteSelect());
    fireEvent.click(
      await screen.findByRole("option", {
        name: "ChatGPT · maria@example.com",
      }),
    );

    await waitFor(() =>
      expect(patchBody).toEqual({
        auth_provider: "codex",
        credential_id: "cred-1",
      }),
    );
    await waitFor(() =>
      expect(
        within(
          screen.getByRole("region", { name: "Maria AI connection" }),
        ).getByRole("combobox").textContent,
      ).toContain("ChatGPT"),
    );
    expect(
      await screen.findByText(
        "New threads, routines and follow-ups run on ChatGPT.",
      ),
    ).toBeDefined();
  });

  it("returns the expert to the account default", async () => {
    let patchBody: unknown;
    server.use(
      getGetExpertMockHandler(mariaOnChatGPT),
      getUpdateExpertLlmRouteMockHandler(async ({ request }) => {
        patchBody = await request.json();
        return maria;
      }),
    );
    render(<ExpertDetailPage />);

    await openTab("Settings");
    fireEvent.click(await findRouteSelect());
    fireEvent.click(
      await screen.findByRole("option", { name: "Account default" }),
    );

    await waitFor(() => expect(patchBody).toEqual({ auth_provider: null }));
  });

  it("tells the owner chat on a subscription does not count against the budget", async () => {
    server.use(getGetExpertMockHandler(mariaOnChatGPT));
    render(<ExpertDetailPage />);

    const budget = await screen.findByRole("region", { name: "Maria budget" });
    expect(
      await within(budget).findByText(/Chat is not metered on ChatGPT/),
    ).toBeDefined();
  });

  it("keeps the budget quiet about metering on the platform route", async () => {
    render(<ExpertDetailPage />);

    const budget = await screen.findByRole("region", { name: "Maria budget" });
    await openTab("Settings");
    await findRouteSelect();
    expect(within(budget).queryByText(/not metered/)).toBeNull();
  });

  it("keeps the budget quiet about metering when the pinned connection is gone", async () => {
    server.use(
      getGetExpertMockHandler({
        ...mariaOnChatGPT,
        llm_credential_id: "cred-gone",
        llm_route_available: false,
      }),
    );
    render(<ExpertDetailPage />);

    const budget = await screen.findByRole("region", { name: "Maria budget" });
    await openTab("Settings");
    await findRouteSelect();
    expect(within(budget).queryByText(/not metered/)).toBeNull();
  });

  it("warns when the pinned connection has gone missing", async () => {
    server.use(
      getGetExpertMockHandler({
        ...mariaOnChatGPT,
        llm_credential_id: "cred-gone",
        llm_route_available: false,
      }),
    );
    render(<ExpertDetailPage />);

    await openTab("Settings");
    const section = await findRouteSection();
    expect(
      await within(section).findByText(/ChatGPT connection missing/),
    ).toBeDefined();
    expect(within(section).getByRole("combobox").textContent).toContain(
      "ChatGPT · not connected",
    );
  });

  it("warns that Microsoft 365 Copilot cannot do unattended tool work", async () => {
    server.use(
      getGetExpertMockHandler({
        ...maria,
        llm_auth_provider: "microsoft_365_copilot",
        llm_credential_id: "cred-msft",
        llm_route_label: "Microsoft 365 Copilot",
      }),
    );
    render(<ExpertDetailPage />);

    await openTab("Settings");
    const section = await findRouteSection();
    expect(await within(section).findByText(/cannot run tools/)).toBeDefined();
  });

  it("lives in the Settings tab rather than Basics", async () => {
    render(<ExpertDetailPage />);

    const basics = await screen.findByRole("tabpanel", { name: "Basics" });
    expect(
      within(basics).queryByRole("region", { name: "Maria AI connection" }),
    ).toBeNull();

    await openTab("Settings");
    const settings = await screen.findByRole("tabpanel", { name: "Settings" });
    expect(
      await within(settings).findByRole("region", {
        name: "Maria AI connection",
      }),
    ).toBeDefined();
  });
});
