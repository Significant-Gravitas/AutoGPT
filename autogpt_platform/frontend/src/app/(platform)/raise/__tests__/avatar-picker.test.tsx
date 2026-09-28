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

async function openPicker() {
  seedAtAvatar();
  render(
    <>
      <RaisePage />
      <Toaster />
    </>,
  );
  await screen.findByRole(
    "group",
    { name: "Managed looks" },
    { timeout: 5000 },
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

test("starts on the General fallback and saves it only after confirmation", async () => {
  await openPicker();
  const general = "/autogpt-characters/v2.1/expert-general-01/neutral/128.webp";
  await userEvent.click(screen.getByRole("button", { name: "General" }));
  expect(loadDraft().avatarUrl).toBeNull();
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  await waitFor(() => expect(loadDraft().avatarUrl).toBe(general));
  expect(loadDraft().color).toBe("violet-300");
  expect(screen.queryByRole("group", { name: "Managed looks" })).toBeNull();
});

test("offers generation and uploads, never another Expert's face or a shade or accent choice", async () => {
  await openPicker();
  expect(
    screen.getByRole("button", { name: "Generate with AI" }),
  ).toBeDefined();
  expect(
    screen.getByRole("button", { name: "Upload a picture" }),
  ).toBeDefined();
  const looks = screen.getByRole("group", { name: "Managed looks" });
  expect(looks.querySelectorAll("button")).toHaveLength(1);
  expect(within(looks).queryByRole("button", { name: "Maria" })).toBeNull();
  expect(screen.queryByRole("combobox", { name: "Shade" })).toBeNull();
  expect(
    screen.queryByRole("combobox", { name: "Accent placement" }),
  ).toBeNull();
  expect(screen.queryByRole("button", { name: "Next hair" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Lavender" })).toBeNull();
});
