import {
  getGetExpertActivityMockHandler,
  getGetExpertMockHandler,
  getGetPublishedTemplateMockHandler,
  getListExpertRunsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

const { portabilityFlag, authUser } = vi.hoisted(() => ({
  portabilityFlag: { enabled: true, ready: true },
  authUser: {
    current: { id: "user-1", role: "admin" } as {
      id: string;
      role: string;
    } | null,
  },
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

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: authUser.current,
    isLoggedIn: Boolean(authUser.current),
    isUserLoading: false,
  }),
}));

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
  bio: "A senior marketing strategist.",
  skills: ["Content strategy"],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [
    {
      id: "wf-1",
      store_listing_version_id: "slv-1",
      library_agent_id: "lib-1",
      graph_id: "graph-1",
      name: "Calendar",
      description: "Plans a week of posts",
    },
  ],
};

// Attached from the library and published afterwards: publishing an agent never
// writes back to the workflow, so the stored listing stays empty even though
// the publish route will find the listing by the agent's graph.
const withUnlistedAgent: Expert = {
  ...maria,
  workflows: [
    {
      ...maria.workflows[0],
      store_listing_version_id: null,
    },
  ],
};

function renderPage() {
  return render(
    <>
      <ExpertDetailPage />
      <Toaster />
    </>,
  );
}

async function openSettings() {
  await userEvent.click(await screen.findByRole("tab", { name: "Settings" }));
}

async function publish() {
  await openSettings();
  await userEvent.click(screen.getByTestId("expert-publish-button"));
  await userEvent.click(await screen.findByRole("button", { name: "Publish" }));
}

beforeEach(() => {
  portabilityFlag.enabled = true;
  authUser.current = { id: "user-1", role: "admin" };
  server.use(
    getGetExpertMockHandler(maria),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertRunsMockHandler([]),
    getGetExpertActivityMockHandler({ timezone: "UTC", days: [] }),
    getGetPublishedTemplateMockHandler(null),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("Publishing an expert to the marketplace", () => {
  test("publishes after the review dialog confirms", async () => {
    server.use(
      http.post("/api/proxy/api/experts/expert-maria/publish", () =>
        HttpResponse.json({ ...maria, id: "template-maria" }, { status: 201 }),
      ),
    );

    renderPage();
    await publish();

    expect(await screen.findByText("Published Maria")).toBeDefined();
  });

  test("publishes an agent that was listed after it was attached", async () => {
    let publishRequests = 0;
    server.use(
      getGetExpertMockHandler(withUnlistedAgent),
      http.post("/api/proxy/api/experts/expert-maria/publish", () => {
        publishRequests += 1;
        return HttpResponse.json(
          { ...withUnlistedAgent, id: "template-maria" },
          { status: 201 },
        );
      }),
    );

    renderPage();
    await openSettings();
    await userEvent.click(screen.getByTestId("expert-publish-button"));

    expect(await screen.findByText("Your agent")).toBeDefined();
    expect(screen.queryByText(/Publish these agents/)).toBeNull();
    const confirm = screen.getByRole("button", { name: "Publish" });
    expect(confirm.hasAttribute("disabled")).toBe(false);

    await userEvent.click(confirm);

    expect(await screen.findByText("Published Maria")).toBeDefined();
    expect(publishRequests).toBe(1);
  });

  test("says the expert is already on the marketplace", async () => {
    server.use(
      getGetPublishedTemplateMockHandler({ ...maria, id: "template-maria" }),
    );

    renderPage();
    await openSettings();

    expect(await screen.findByText("Live on marketplace")).toBeDefined();
    expect(screen.getByTestId("expert-publish-button")).toHaveProperty(
      "textContent",
      "Publish again",
    );
  });

  test("names the agents the backend refuses to publish without", async () => {
    server.use(
      getGetExpertMockHandler(withUnlistedAgent),
      http.post("/api/proxy/api/experts/expert-maria/publish", () =>
        HttpResponse.json(
          {
            detail: {
              code: "unpublished_workflows",
              workflows: ["Calendar"],
              message: "Publish this agent first",
            },
          },
          { status: 400 },
        ),
      ),
    );

    renderPage();
    await publish();

    expect(await screen.findByText("Couldn't publish Maria")).toBeDefined();
    expect(
      screen.getByText(
        "Publish these agents to the marketplace first: Calendar",
      ),
    ).toBeDefined();
  });

  test("says when the session is not allowed to publish", async () => {
    server.use(
      http.post("/api/proxy/api/experts/expert-maria/publish", () =>
        HttpResponse.json({ detail: "Forbidden" }, { status: 403 }),
      ),
    );

    renderPage();
    await publish();

    expect(await screen.findByText("Only admins can publish")).toBeDefined();
  });

  test("hides the row from non-admins and while the flag is off", async () => {
    authUser.current = { id: "user-1", role: "authenticated" };

    const view = renderPage();
    await openSettings();
    expect(screen.queryByTestId("expert-publish-button")).toBeNull();
    view.unmount();

    authUser.current = { id: "user-1", role: "admin" };
    portabilityFlag.enabled = false;
    renderPage();
    await openSettings();
    expect(screen.queryByTestId("expert-publish-button")).toBeNull();
  });
});
