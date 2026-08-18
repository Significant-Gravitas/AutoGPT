import { Toaster } from "@/components/molecules/Toast/toaster";
import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { saveDraft } from "../helpers";
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

function renderRaise() {
  return render(
    <>
      <RaisePage />
      <Toaster />
    </>,
  );
}

function visibleBubbleText() {
  const log = screen.getByRole("log", { name: "Raise expert conversation" });
  return Array.from(log.querySelectorAll("span[aria-hidden]")).map(
    (node) => node.textContent ?? "",
  );
}

beforeEach(() => {
  window.sessionStorage.clear();
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
});

afterEach(() => {
  vi.clearAllMocks();
});

test("renders restored conversation messages instantly after reload", async () => {
  saveDraft({
    step: "color",
    hasStarted: true,
    role: "marketer",
    name: "Nova",
    color: null,
    avatarUrl: null,
    about: null,
    voicePreferences: "",
    voiceLabel: null,
    budget: null,
    marketplace: null,
    skills: null,
  });

  renderRaise();

  expect(
    await screen.findByRole(
      "log",
      { name: "Raise expert conversation" },
      { timeout: 5000 },
    ),
  ).toBeDefined();

  const visible = visibleBubbleText();
  expect(
    visible.some((text) =>
      text.includes(
        "Hello, I'm Autopilot. I'll help you raise your own expert.",
      ),
    ),
  ).toBe(true);
  expect(
    visible.some((text) =>
      text.includes("First — what should your expert do for you?"),
    ),
  ).toBe(true);
  expect(
    visible.some((text) =>
      text.includes("Good pick. What do you want to call it?"),
    ),
  ).toBe(true);
  expect(
    visible.some((text) => text.includes("Nice. Now choose a color for it.")),
  ).toBe(true);
});
