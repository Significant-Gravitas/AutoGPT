import { getGetV2ListSessionsMockHandler } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ListSessionsResponse } from "@/app/api/__generated__/models/listSessionsResponse";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { NamingMomentCard } from "../NamingMomentCard";

const { authUserMock, setFlagStatusMock } = vi.hoisted(() => ({
  authUserMock: { current: { id: "user-1" } },
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
  useAuth: () => ({ isLoggedIn: true, user: authUserMock.current }),
}));

const { pushMock } = vi.hoisted(() => ({ pushMock: vi.fn() }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

const anExpert = {
  id: "expert-1",
  name: "Otto",
  avatar_url: null,
  role: "",
  is_template: false,
  source_template_id: null,
  is_archived: false,
} as Expert;

const aSession = {
  id: "session-1",
  created_at: "2026-08-14T00:00:00Z",
  updated_at: "2026-08-14T00:00:00Z",
  is_processing: false,
} as SessionSummaryResponse;

function useHandlers({ experts, total }: { experts: Expert[]; total: number }) {
  const sessions: ListSessionsResponse = {
    sessions: total > 0 ? [aSession] : [],
    total,
  };
  server.use(
    getListExpertsMockHandler(experts),
    getGetV2ListSessionsMockHandler(sessions),
  );
}

beforeEach(() => {
  authUserMock.current = { id: "user-1" };
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  pushMock.mockClear();
  window.localStorage.clear();
});

afterEach(() => {
  vi.clearAllMocks();
});

test("shows for an existing user with sessions and no experts", async () => {
  useHandlers({ experts: [], total: 3 });

  render(<NamingMomentCard />);

  expect(
    await screen.findByRole("button", { name: "Give me a name" }),
  ).toBeTruthy();
});

test("does not show when the user already has an expert", async () => {
  useHandlers({ experts: [anExpert], total: 3 });

  render(<NamingMomentCard />);

  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Give me a name" })).toBeNull(),
  );
});

test("does not show for a fresh user with no sessions", async () => {
  useHandlers({ experts: [], total: 0 });

  render(<NamingMomentCard />);

  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Give me a name" })).toBeNull(),
  );
});

test("does not show when the experts flag is off", async () => {
  setFlagStatusMock.mockReturnValue({ enabled: false, ready: true });
  useHandlers({ experts: [], total: 3 });

  render(<NamingMomentCard />);

  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Give me a name" })).toBeNull(),
  );
});

test("'Give me a name' routes to the naming raise flow", async () => {
  useHandlers({ experts: [], total: 3 });

  render(<NamingMomentCard />);

  await userEvent.click(
    await screen.findByRole("button", { name: "Give me a name" }),
  );

  expect(pushMock).toHaveBeenCalledWith("/raise?from=naming");
});

test("'Not now' dismisses the card and persists the dismissal", async () => {
  useHandlers({ experts: [], total: 3 });

  const { unmount } = render(<NamingMomentCard />);

  await userEvent.click(await screen.findByRole("button", { name: "Not now" }));

  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Not now" })).toBeNull(),
  );

  unmount();
  render(<NamingMomentCard />);

  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Give me a name" })).toBeNull(),
  );
});

test("an account change resets dismissal state for the current user", async () => {
  window.localStorage.setItem("autogpt:naming-moment-dismissed:user-1", "true");
  useHandlers({ experts: [], total: 3 });

  const { rerender } = render(<NamingMomentCard />);

  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Give me a name" })).toBeNull(),
  );

  authUserMock.current = { id: "user-2" };
  rerender(<NamingMomentCard />);

  expect(
    await screen.findByRole("button", { name: "Give me a name" }),
  ).toBeTruthy();
});

test("a dismissed user fires no experts or sessions requests on mount", async () => {
  window.localStorage.setItem("autogpt:naming-moment-dismissed:user-1", "true");
  let expertsRequests = 0;
  let sessionsRequests = 0;
  server.use(
    getListExpertsMockHandler(() => {
      expertsRequests += 1;
      return [];
    }),
    getGetV2ListSessionsMockHandler(() => {
      sessionsRequests += 1;
      return { sessions: [aSession], total: 3 };
    }),
  );

  render(<NamingMomentCard />);

  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Give me a name" })).toBeNull(),
  );
  // The card unrenders synchronously either way; give a wrongly-enabled query
  // enough event-loop turns to reach the mock server before asserting silence.
  await new Promise((resolve) => setTimeout(resolve, 100));
  expect(expertsRequests).toBe(0);
  expect(sessionsRequests).toBe(0);
});
