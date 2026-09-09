import {
  getListExpertCredentialsMockHandler,
  getListExpertPodsMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV2ListLibraryAgentsMockHandler200 } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import TeamPage from "../page";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
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

function makeExpert(over: Partial<Expert> = {}): Expert {
  return {
    id: "expert-maria",
    name: "Maria",
    avatar_url: null,
    role: "Marketing Strategist",
    bio: null,
    skills: [],
    tagline: "Grows your brand while you sleep",
    identity: "You are Maria.",
    voice_preferences: null,
    boundaries: null,
    protected_soul_rules: [],
    is_template: false,
    source_template_id: "template-maria",
    is_archived: false,
    workflows: [],
    ...over,
  } as Expert;
}

const lee = makeExpert({ id: "expert-lee", name: "Lee", role: "Researcher" });

beforeEach(() => {
  server.use(
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertPodsMockHandler([]),
    getListExpertCredentialsMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200(),
  );
});

describe("Pods are hidden", () => {
  test("keeps experts visible without pod UI even when pods exist", async () => {
    const growth: ExpertPod = {
      id: "pod-growth",
      name: "Growth",
      created_at: new Date("2026-08-14T00:00:00Z"),
    };
    server.use(
      getListExpertsMockHandler([makeExpert({ pod_id: growth.id }), lee]),
      getListExpertPodsMockHandler([growth]),
    );

    render(<TeamPage />);

    expect(await screen.findByText("Maria")).toBeDefined();
    expect(screen.getByText("Lee")).toBeDefined();
    expect(screen.getByRole("tab", { name: "Team Overview" })).toBeDefined();
    expect(screen.getByRole("link", { name: "Raise expert" })).toBeDefined();
    expect(screen.getByRole("link", { name: "Hire expert" })).toBeDefined();
    expect(
      screen.queryByRole("tab", { name: "Pod board", hidden: true }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: /pod/i, hidden: true }),
    ).toBeNull();
    expect(
      screen.queryByRole("region", { name: "Pods", hidden: true }),
    ).toBeNull();
    expect(
      screen.queryByRole("dialog", { name: /pod/i, hidden: true }),
    ).toBeNull();
    expect(screen.queryByText("Growth")).toBeNull();
  });

  test("does not offer pod creation for an empty team", async () => {
    server.use(getListExpertsMockHandler([]));

    render(<TeamPage />);

    expect(
      await screen.findByRole("link", { name: "Browse the marketplace" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("tab", { name: "Pod board", hidden: true }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: /pod/i, hidden: true }),
    ).toBeNull();
    expect(screen.queryByText("No pods yet")).toBeNull();
  });
});
