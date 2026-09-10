import {
  getListExpertCredentialsMockHandler,
  getListExpertPodsMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV2ListLibraryAgentsMockHandler200 } from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getForgetMyExpertMemoryFactMockHandler200,
  getListMyExpertMemoryFactsMockHandler,
  getListMyExpertMemoryFactsMockHandler200,
} from "@/app/api/__generated__/endpoints/memory/memory.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { MemoryFact } from "@/app/api/__generated__/models/memoryFact";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

const { memoryFlagMock } = vi.hoisted(() => ({
  memoryFlagMock: vi.fn(() => true),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "graphiti-memory"
        ? memoryFlagMock()
        : actual.useGetFlag(flag as never),
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? { enabled: true, ready: true }
        : actual.useFlagStatus(flag as never),
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
  usePathname: () => "/team",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: vi.fn(),
}));

const maria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

function makeFact(over: Partial<MemoryFact> = {}): MemoryFact {
  return {
    uuid: "edge-1",
    fact: "Q4 campaign brief is due Friday",
    name: "due",
    source: "Campaign",
    target: "Friday",
    created_at: "2026-08-17T00:00:00Z",
    ...over,
  } as MemoryFact;
}

async function openSoulDrawer() {
  const user = userEvent.setup();
  render(<TeamPage />);
  await user.click(await screen.findByRole("button", { name: "Edit Soul" }));
  return user;
}

beforeEach(() => {
  memoryFlagMock.mockReturnValue(true);
  server.use(
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertPodsMockHandler([]),
    getListExpertCredentialsMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200(),
    getListExpertsMockHandler([maria]),
  );
});

describe("Soul drawer — what I've learned", () => {
  test("lists the expert's own memories, newest first, with a link to the rest", async () => {
    const factRequests: string[] = [];
    server.use(
      getListMyExpertMemoryFactsMockHandler(async (info) => {
        factRequests.push(String(info.params.expertId));
        return {
          expert_id: "expert-maria",
          items: [
            makeFact(),
            makeFact({
              uuid: "edge-2",
              fact: "Emberline ships to the EU only",
              created_at: "2026-08-16T00:00:00Z",
            }),
          ],
        };
      }),
    );

    await openSoulDrawer();

    expect(
      await screen.findByText("Q4 campaign brief is due Friday"),
    ).toBeDefined();
    expect(screen.getByText("Emberline ships to the EU only")).toBeDefined();
    expect(screen.queryByText(/Nothing recorded yet/)).toBeNull();

    // Scoped to this expert, and only this expert.
    expect(factRequests).toEqual(["expert-maria"]);

    const seeAll = screen.getByRole("link", {
      name: "See all in memory settings",
    });
    expect(seeAll.getAttribute("href")).toBe(
      "/settings/memory?expert=expert-maria",
    );
  });

  test("keeps the empty state for an expert that has learned nothing", async () => {
    server.use(
      getListMyExpertMemoryFactsMockHandler200({
        expert_id: "expert-maria",
        items: [],
      }),
    );

    await openSoulDrawer();

    expect(
      await screen.findByText(
        "Nothing recorded yet. What this expert learns will appear here.",
      ),
    ).toBeDefined();
    expect(
      screen.queryByRole("link", { name: "See all in memory settings" }),
    ).toBeNull();
  });

  test("forgetting a memory calls the scoped delete and refetches", async () => {
    const forgotten: string[] = [];
    let listCalls = 0;
    server.use(
      getListMyExpertMemoryFactsMockHandler(async () => {
        listCalls += 1;
        return {
          expert_id: "expert-maria",
          items: forgotten.length ? [] : [makeFact()],
        };
      }),
      getForgetMyExpertMemoryFactMockHandler200(async (info) => {
        forgotten.push(String(info.params.factUuid));
        return { uuid: String(info.params.factUuid), forgotten: true };
      }),
    );

    const user = await openSoulDrawer();

    await screen.findByText("Q4 campaign brief is due Friday");
    const callsBeforeForget = listCalls;
    await user.click(screen.getByRole("button", { name: "Forget" }));

    await waitFor(() => expect(forgotten).toEqual(["edge-1"]));
    await waitFor(() => expect(listCalls).toBeGreaterThan(callsBeforeForget));
    await waitFor(() =>
      expect(
        screen.getByText(
          "Nothing recorded yet. What this expert learns will appear here.",
        ),
      ).toBeDefined(),
    );
  });

  test("says so when the memories cannot be loaded, instead of claiming there are none", async () => {
    server.use(
      http.get(
        "/api/proxy/api/memory/experts/:expertId/facts",
        () => new HttpResponse(null, { status: 500 }),
      ),
    );

    await openSoulDrawer();

    expect(
      await screen.findByText("Couldn't load what this expert has learned."),
    ).toBeDefined();
    expect(screen.queryByText(/Nothing recorded yet/)).toBeNull();
  });

  test("asks the memory API for nothing while the memory flag is off", async () => {
    memoryFlagMock.mockReturnValue(false);
    let listCalls = 0;
    server.use(
      getListMyExpertMemoryFactsMockHandler(async () => {
        listCalls += 1;
        return { expert_id: "expert-maria", items: [makeFact()] };
      }),
    );

    await openSoulDrawer();

    expect(
      await screen.findByText(
        "Nothing recorded yet. What this expert learns will appear here.",
      ),
    ).toBeDefined();
    expect(listCalls).toBe(0);
  });
});
