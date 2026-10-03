import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { colorForCategory } from "../components/CategoryStep/helpers";
import { draftWithPrefilledRole, EMPTY_DRAFT, saveDraft } from "../helpers";
import RaisePage from "../page";

const { searchParams } = vi.hoisted(() => ({
  searchParams: { current: new URLSearchParams() },
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
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

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, user: { id: "user-1" } }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/raise",
  useSearchParams: () => searchParams.current,
  useParams: () => ({}),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

// The conversation types its questions in; reduced motion reveals each
// beat immediately so the name step is queryable.
function mockReducedMotion() {
  vi.spyOn(window, "matchMedia").mockImplementation(
    (query) =>
      ({
        matches: query.includes("prefers-reduced-motion"),
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }) as unknown as MediaQueryList,
  );
}

beforeEach(() => {
  window.sessionStorage.clear();
  mockReducedMotion();
  searchParams.current = new URLSearchParams();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("/raise?role= — the greeting page's raise door", () => {
  it("opens on the job title question with the area already answered", async () => {
    searchParams.current = new URLSearchParams("role=support");

    render(<RaisePage />);

    expect(
      await screen.findByRole(
        "group",
        { name: "Suggested job titles" },
        { timeout: 5000 },
      ),
    ).toBeDefined();
    // Support's job titles, so the role's area really did land in the draft.
    expect(
      screen.getByRole("button", { name: "Customer Success Manager" }),
    ).toBeDefined();
  });

  it("leaves a draft in progress alone", () => {
    const inProgress = {
      ...EMPTY_DRAFT,
      hasStarted: true,
      category: "marketing" as const,
      color: "orange-300",
      jobTitle: "Marketing Manager",
      name: "Nova",
      step: "avatar" as const,
    };
    saveDraft(inProgress);

    expect(draftWithPrefilledRole(inProgress, "support")).toEqual(inProgress);
  });

  it("ignores a role outside the wizard's role ids", () => {
    expect(draftWithPrefilledRole(EMPTY_DRAFT, "")).toEqual(EMPTY_DRAFT);
    expect(draftWithPrefilledRole(EMPTY_DRAFT, "Chief of staff")).toEqual(
      EMPTY_DRAFT,
    );
  });

  it("answers the area and its color from the role", () => {
    expect(draftWithPrefilledRole(EMPTY_DRAFT, "analyst")).toEqual({
      ...EMPTY_DRAFT,
      hasStarted: true,
      category: "finance",
      color: colorForCategory("finance"),
      step: "jobTitle",
    });
  });
});
