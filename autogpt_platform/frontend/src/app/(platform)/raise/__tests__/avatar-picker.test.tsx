import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { notionConfigForName } from "@/components/molecules/NotionAvatar/helpers";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { loadDraft, saveDraft, VOICE_SKIPPED_LABEL } from "../helpers";
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

function seedAtAvatar(name = "Maria") {
  saveDraft({
    step: "avatar",
    hasStarted: true,
    role: "marketer",
    name,
    color: "violet-300",
    avatarUrl: null,
    about: null,
    voicePreferences: "",
    voiceLabel: VOICE_SKIPPED_LABEL,
    budget: null,
    marketplace: null,
    skills: null,
  });
}

function drawnAvatar() {
  return screen.getByTestId("notion-avatar").getAttribute("data-avatar");
}

async function openGenerator() {
  seedAtAvatar();
  render(
    <>
      <RaisePage />
      <Toaster />
    </>,
  );
  await userEvent.click(
    await screen.findByRole(
      "button",
      { name: "Generate a face" },
      { timeout: 5000 },
    ),
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

test("opens on the face the expert's name seeds, in the colour they were given", async () => {
  await openGenerator();

  const seeded = notionConfigForName("Maria");
  const slots = drawnAvatar()?.split(".")[0];

  expect(slots).toBe(
    [
      seeded.parts.face,
      seeded.parts.nose,
      seeded.parts.mouth,
      seeded.parts.eyes,
      seeded.parts.eyebrows,
      seeded.parts.glasses,
      seeded.parts.hair,
      seeded.parts.accessories,
      seeded.parts.details,
      seeded.parts.beard,
    ].join("-"),
  );
  // The colour was answered a beat earlier, so it carries into the face.
  expect(drawnAvatar()).toContain(".lavender");
});

test("shuffle changes the face but keeps the chosen colour", async () => {
  await openGenerator();
  const before = drawnAvatar();

  await userEvent.click(screen.getByRole("button", { name: "Shuffle" }));

  await waitFor(() => expect(drawnAvatar()).not.toBe(before));
  expect(drawnAvatar()).toContain(".lavender");
});

test("cycling a feature moves only that feature", async () => {
  await openGenerator();
  const before = drawnAvatar()!.split(".")[0].split("-");

  await userEvent.click(screen.getByRole("button", { name: "Next hair" }));

  await waitFor(() => {
    const after = drawnAvatar()!.split(".")[0].split("-");
    const moved = after.filter((slot, index) => slot !== before[index]);
    expect(moved).toHaveLength(1);
    // Hair is the seventh slot in draw order.
    expect(after[6]).not.toBe(before[6]);
  });
});

test("cycling backwards from the first part wraps to the last", async () => {
  await openGenerator();

  await userEvent.click(screen.getByRole("button", { name: "Previous eyes" }));
  const eyes = Number(drawnAvatar()!.split(".")[0].split("-")[3]);

  expect(eyes).toBeGreaterThan(0);
});

test("the picked face becomes the draft's avatar and closes the picker", async () => {
  await openGenerator();
  await userEvent.click(screen.getByRole("button", { name: "Next hair" }));
  const picked = drawnAvatar();

  await userEvent.click(screen.getByRole("button", { name: "Use this face" }));

  await waitFor(() =>
    expect(loadDraft().avatarUrl).toBe(`/avatars/notion/${picked}.svg`),
  );
  expect(screen.queryByRole("button", { name: "Shuffle" })).toBeNull();
});

test("cancelling the picker leaves the avatar unanswered", async () => {
  await openGenerator();

  await userEvent.click(screen.getByRole("button", { name: "Cancel" }));

  expect(
    await screen.findByRole("button", { name: "Generate a face" }),
  ).toBeDefined();
  expect(loadDraft().avatarUrl).toBeNull();
});
