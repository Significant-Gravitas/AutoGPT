import {
  getGetExpertMockHandler,
  getListExpertCredentialsMockHandler,
  getListExpertRunsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ExpertDetailPage from "../page";

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
  weekly_budget: null,
  weekly_spend: 0,
  schedules_paused_at: null,
  pod_id: null,
} as unknown as Expert;

const linkedin: ExpertCredentialRef = {
  credential_id: "cred-linkedin",
  provider: "linkedin",
  title: "Work LinkedIn",
  type: "oauth2",
};

beforeEach(() => {
  server.use(
    getGetExpertMockHandler(maria),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertRunsMockHandler([]),
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json([
        {
          id: "cred-linkedin",
          provider: "linkedin",
          type: "oauth2",
          title: "Work LinkedIn",
        },
        {
          id: "cred-notion",
          provider: "notion",
          type: "api_key",
          title: "Team Notion",
        },
      ]),
    ),
  );
});

describe("managing an expert's integrations", () => {
  it("lists what the expert can reach", async () => {
    server.use(getListExpertCredentialsMockHandler([linkedin]));

    render(<ExpertDetailPage />);

    const section = await screen.findByTestId("expert-integrations-section");
    expect(await within(section).findByText("Work LinkedIn")).toBeDefined();
    expect(within(section).getByText("LinkedIn")).toBeDefined();
  });

  it("explains the empty state instead of showing a bare list", async () => {
    server.use(getListExpertCredentialsMockHandler([]));

    render(<ExpertDetailPage />);

    const section = await screen.findByTestId("expert-integrations-section");
    expect(
      await within(section).findByText(
        /Maria cannot reach any of your integrations/,
      ),
    ).toBeDefined();
  });

  it("revokes an integration through the API", async () => {
    let revoked: string | null = null;
    server.use(
      getListExpertCredentialsMockHandler([linkedin]),
      http.delete(
        "*/api/experts/expert-maria/credentials/:credentialId",
        ({ params }) => {
          revoked = params.credentialId as string;
          return HttpResponse.json([]);
        },
      ),
    );

    render(<ExpertDetailPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Remove Work LinkedIn" }),
    );

    await waitFor(() => expect(revoked).toBe("cred-linkedin"));
  });

  it("only offers credentials the expert does not already have", async () => {
    let granted: string[] = [];
    server.use(
      getListExpertCredentialsMockHandler([linkedin]),
      http.post(
        "*/api/experts/expert-maria/credentials",
        async ({ request }) => {
          const body = (await request.json()) as { credential_ids: string[] };
          granted = body.credential_ids;
          return HttpResponse.json([linkedin]);
        },
      ),
    );

    render(<ExpertDetailPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );

    expect(await screen.findByText("Team Notion")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Work LinkedIn" })).toBeNull();

    await userEvent.click(screen.getByText("Team Notion"));
    await waitFor(() => expect(granted).toEqual(["cred-notion"]));
  });

  it("offers connecting a new service when there is nothing left to grant", async () => {
    server.use(
      getListExpertCredentialsMockHandler([]),
      http.get("*/api/integrations/providers", () => HttpResponse.json([])),
      http.get("*/api/integrations/credentials", () => HttpResponse.json([])),
    );

    render(<ExpertDetailPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );
    expect(
      await screen.findByText("You haven't connected any services yet."),
    ).toBeDefined();

    await userEvent.click(
      screen.getByRole("button", { name: /Connect a new service/ }),
    );

    expect(await screen.findByLabelText("Search services")).toBeDefined();
  });

  it("grants a credential connected from inside the dialog", async () => {
    const workLinkedin = {
      id: "cred-linkedin",
      provider: "linkedin",
      type: "oauth2",
      title: "Work LinkedIn",
    };
    const teamSlack = {
      id: "cred-slack",
      provider: "slack",
      type: "oauth2",
      title: "Team Slack",
    };
    let connected = [workLinkedin];
    let granted: string[] = [];

    server.use(
      getListExpertCredentialsMockHandler([linkedin]),
      http.get("*/api/integrations/providers", () => HttpResponse.json([])),
      http.get("*/api/integrations/credentials", () =>
        HttpResponse.json(connected),
      ),
      http.post(
        "*/api/experts/expert-maria/credentials",
        async ({ request }) => {
          const body = (await request.json()) as { credential_ids: string[] };
          granted = body.credential_ids;
          return HttpResponse.json([linkedin]);
        },
      ),
    );

    render(<ExpertDetailPage />);

    await userEvent.click(
      await screen.findByRole("button", { name: /Add integration/ }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: /Connect a new service/ }),
    );
    await screen.findByLabelText("Search services");

    connected = [workLinkedin, teamSlack];
    await userEvent.click(screen.getByRole("button", { name: "Close" }));

    await waitFor(() => expect(granted).toEqual(["cred-slack"]));
  });
});
