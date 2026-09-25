import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
  EMPTY_DRAFT,
  loadDraft,
  saveDraft,
  VOICE_SKIPPED_LABEL,
} from "../helpers";
import RaisePage from "../page";

const { setFlagStatusMock } = vi.hoisted(() => ({
  setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
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
        ? setFlagStatusMock()
        : actual.useFlagStatus(flag as never),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, user: { id: "user-1" } }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/raise",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

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

function generationHandlers(requests: unknown[] = []) {
  return [
    http.post("*/api/experts/avatars/generations", async ({ request }) => {
      requests.push(await request.json());
      return HttpResponse.json(
        { id: "job-1", status: "pending" },
        { status: 202 },
      );
    }),
    http.get("*/api/experts/avatars/generations/job-1", () =>
      HttpResponse.json({
        id: "job-1",
        status: "complete",
        avatar_url: "https://cdn.test/generated.png",
      }),
    ),
  ];
}

function seedAt(
  step: "category" | "avatar",
  category: ExpertAvatarRequestCategory | null = null,
) {
  saveDraft({
    ...EMPTY_DRAFT,
    step,
    hasStarted: true,
    role: "marketer",
    jobTitle: "Marketing Manager",
    name: "Maria",
    category,
    color: category ? "green-300" : null,
    voiceLabel: VOICE_SKIPPED_LABEL,
  });
}

function renderRaise() {
  render(
    <>
      <RaisePage />
      <Toaster />
    </>,
  );
}

beforeEach(() => {
  window.sessionStorage.clear();
  mockReducedMotion();
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  server.use(getListCopilotSkillsMockHandler([]));
});

afterEach(() => {
  vi.clearAllMocks();
});

test("the category beat answers the color before any avatar is drawn", async () => {
  const requests: unknown[] = [];
  server.use(...generationHandlers(requests));
  seedAt("category");
  renderRaise();

  const categories = await screen.findByRole("group", {
    name: "What the expert works on",
  });
  expect(requests).toHaveLength(0);
  await userEvent.click(
    await within(categories).findByRole("button", { name: "Finance" }),
  );

  await waitFor(() => expect(loadDraft().category).toBe("finance"));
  expect(loadDraft().color).toBe("green-300");
  expect(loadDraft().avatarUrl).toBeNull();

  await waitFor(() => expect(requests).toHaveLength(1));
  expect(requests[0]).toMatchObject({ category: "finance" });
});

test("the generated avatar is saved only after confirmation", async () => {
  server.use(...generationHandlers());
  seedAt("avatar", "finance");
  renderRaise();

  await waitFor(() =>
    expect(
      screen.getByRole("img", { name: "Maria" }).getAttribute("src"),
    ).toContain("generated.png"),
  );
  expect(loadDraft().avatarUrl).toBeNull();

  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  await waitFor(() =>
    expect(loadDraft().avatarUrl).toBe("https://cdn.test/generated.png"),
  );
  expect(loadDraft().color).toBe("green-300");
});

test("regenerating asks for another avatar in the same category", async () => {
  const requests: unknown[] = [];
  server.use(...generationHandlers(requests));
  seedAt("avatar", "finance");
  renderRaise();

  await waitFor(() => expect(requests).toHaveLength(1));
  await userEvent.click(
    await screen.findByRole("button", { name: "Regenerate" }),
  );
  await waitFor(() => expect(requests).toHaveLength(2));
  expect(requests[1]).toMatchObject({ category: "finance" });
});

test("the picker offers upload and no face-part controls", async () => {
  server.use(...generationHandlers());
  seedAt("avatar", "finance");
  renderRaise();

  expect(
    await screen.findByRole("button", { name: "Upload a picture" }),
  ).toBeDefined();
  expect(screen.queryByRole("group", { name: "Avatar catalog" })).toBeNull();
  expect(screen.queryByRole("combobox")).toBeNull();
});
