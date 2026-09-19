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

function seedAtAvatar(name = "Maria", color: string | null = "violet-300") {
  saveDraft({
    step: "avatar",
    hasStarted: true,
    role: "marketer",
    jobTitle: "Marketing Manager",
    name,
    color,
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

async function openGenerator(name = "Maria") {
  seedAtAvatar(name);
  render(
    <>
      <RaisePage />
      <Toaster />
    </>,
  );
  // The beat opens straight onto the picker; the artwork is a lazy chunk, so
  // the face itself arrives a tick later.
  await screen.findByTestId("notion-avatar", undefined, { timeout: 5000 });
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
  expect(drawnAvatar()).toContain(".violet");
});

function slotOf(feature: "details" | "hair") {
  // Slots follow the draw order: face, nose, mouth, eyes, eyebrows, glasses,
  // hair, accessories, details, beard.
  const index = feature === "hair" ? 6 : 8;
  return Number(drawnAvatar()!.split(".")[0].split("-")[index]);
}

test("opens with no marks, and shuffling does not add them", async () => {
  await openGenerator();
  expect(slotOf("details")).toBe(0);

  for (let attempt = 0; attempt < 8; attempt += 1) {
    await userEvent.click(screen.getByRole("button", { name: "Shuffle" }));
    expect(slotOf("details")).toBe(0);
  }
});

test("marks are still available from their own row", async () => {
  await openGenerator();

  await userEvent.click(screen.getByRole("button", { name: "Next marks" }));

  await waitFor(() => expect(slotOf("details")).toBe(1));
});

test("shuffle changes the face but keeps the chosen colour", async () => {
  await openGenerator();
  const before = drawnAvatar();

  await userEvent.click(screen.getByRole("button", { name: "Shuffle" }));

  await waitFor(() => expect(drawnAvatar()).not.toBe(before));
  expect(drawnAvatar()).toContain(".violet");
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

test("the picked face and colour become the draft's answer, and the picker closes", async () => {
  await openGenerator();
  await userEvent.click(screen.getByRole("button", { name: "Next hair" }));
  const picked = drawnAvatar();

  await userEvent.click(screen.getByRole("button", { name: "Use this face" }));

  await waitFor(() =>
    expect(loadDraft().avatarUrl).toBe(`/avatars/notion/${picked}.svg`),
  );
  expect(loadDraft().color).toBe("violet-300");
  expect(screen.queryByRole("button", { name: "Shuffle" })).toBeNull();
});

test("the cycle arrows carry no tooltip, which would cover the row above", async () => {
  await openGenerator();

  // The Button atom pops a tooltip for any icon-only button with an
  // aria-label. Stacked this tightly, that tooltip lands on its neighbours.
  // A tooltip trigger leaves data-state on the button it wraps.
  for (const name of ["Next beard", "Previous beard", "Next hair"]) {
    const button = screen.getByRole("button", { name });
    expect(button.getAttribute("data-state")).toBeNull();
    expect(button.getAttribute("aria-describedby")).toBeNull();
  }
});

test("the colour is chosen in the picker and recolours the face", async () => {
  await openGenerator();
  expect(drawnAvatar()).toContain(".violet");

  await userEvent.click(screen.getByRole("button", { name: "Emerald" }));

  await waitFor(() => expect(drawnAvatar()).toContain(".emerald"));

  await userEvent.click(screen.getByRole("button", { name: "Use this face" }));
  await waitFor(() => expect(loadDraft().color).toBe("emerald-300"));
});

test("an expert with no colour yet still opens on a face", async () => {
  seedAtAvatar("Nova", null);
  render(
    <>
      <RaisePage />
      <Toaster />
    </>,
  );

  const avatar = await screen.findByTestId("notion-avatar", undefined, {
    timeout: 5000,
  });
  expect(avatar.getAttribute("data-avatar")).toBeTruthy();
});
