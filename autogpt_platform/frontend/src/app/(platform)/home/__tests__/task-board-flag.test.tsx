import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { Flag } from "@/services/feature-flags/use-get-flag";
import ChatPage from "../../chat/page";
import HomePage from "../page";

const { flagStatusMock, routerReplace, navigation } = vi.hoisted(() => ({
  flagStatusMock: vi.fn(() => ({
    enabled: false,
    ready: true,
    answered: true,
  })),
  routerReplace: vi.fn(),
  navigation: { pathname: "/home" },
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: Flag) =>
      flag === actual.Flag.TASK_BOARD
        ? flagStatusMock()
        : actual.useFlagStatus(flag),
  };
});

vi.mock("@/app/(platform)/copilot/CopilotPage", () => ({
  CopilotPage: () => <div>chat page</div>,
}));

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useRouter: () => ({
      push: vi.fn(),
      replace: routerReplace,
      prefetch: vi.fn(),
    }),
    usePathname: () => navigation.pathname,
    useSearchParams: () => new URLSearchParams(),
  };
});

function setFlag(enabled: boolean, ready = true) {
  flagStatusMock.mockReturnValue({ enabled, ready, answered: ready });
}

beforeEach(() => {
  routerReplace.mockClear();
  window.history.replaceState({}, "", "/");
});

afterEach(() => {
  setFlag(false);
});

describe("/home with the task board flag", () => {
  beforeEach(() => {
    navigation.pathname = "/home";
  });

  test("renders the board when the flag is on", () => {
    setFlag(true);
    render(<HomePage />);

    expect(screen.getByRole("heading", { name: /board/i })).toBeDefined();
    expect(screen.queryByText("chat page")).toBeNull();
  });

  test("renders the chat page when the flag is off", () => {
    setFlag(false);
    render(<HomePage />);

    expect(screen.getByText("chat page")).toBeDefined();
    expect(screen.queryByRole("heading", { name: /board/i })).toBeNull();
  });

  test("renders the chat page until the flag resolves", () => {
    setFlag(true, false);
    render(<HomePage />);

    expect(screen.getByText("chat page")).toBeDefined();
    expect(screen.queryByRole("heading", { name: /board/i })).toBeNull();
  });
});

describe("chat links on /home with the task board flag", () => {
  beforeEach(() => {
    navigation.pathname = "/home";
  });

  test.each([
    ["/home?expertId=e1&kickoff=1", "/chat?expertId=e1&kickoff=1"],
    ["/home?expertId=e1", "/chat?expertId=e1"],
    [
      "/home?source=import&autosubmit=true",
      "/chat?source=import&autosubmit=true",
    ],
    ["/home#prompt=Draft%20a%20plan", "/chat#prompt=Draft%20a%20plan"],
    ["/home?autosubmit=true#prompt=Hi", "/chat?autosubmit=true#prompt=Hi"],
  ])("sends %s to the full-page chat", async (from, to) => {
    setFlag(true);
    window.history.replaceState({}, "", from);
    render(<HomePage />);

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith(to));
    expect(screen.queryByRole("heading", { name: /board/i })).toBeNull();
  });

  test("opens the board for a session link", () => {
    setFlag(true);
    window.history.replaceState(
      {},
      "",
      "/home?sessionId=3f2b8c1e-9a4d-4e6f-8b7a-1c2d3e4f5a6b",
    );
    render(<HomePage />);

    expect(screen.getByRole("heading", { name: /board/i })).toBeDefined();
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("leaves chat links to the chat page when the flag is off", () => {
    setFlag(false);
    window.history.replaceState({}, "", "/home?expertId=e1&kickoff=1");
    render(<HomePage />);

    expect(screen.getByText("chat page")).toBeDefined();
    expect(routerReplace).not.toHaveBeenCalled();
  });
});

describe("/chat with the task board flag", () => {
  beforeEach(() => {
    navigation.pathname = "/chat";
  });

  test("renders the full-page chat when the flag is on", () => {
    setFlag(true);
    render(<ChatPage />);

    expect(screen.getByText("chat page")).toBeDefined();
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("sends the user to /home with the same query when the flag is off", async () => {
    setFlag(false);
    window.history.replaceState({}, "", "/chat?sessionId=abc&expertId=e1");
    render(<ChatPage />);

    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith(
        "/home?sessionId=abc&expertId=e1",
      ),
    );
    expect(screen.queryByText("chat page")).toBeNull();
  });

  test("sends the user to /home when the flag is off and there is no query", async () => {
    setFlag(false);
    window.history.replaceState({}, "", "/chat");
    render(<ChatPage />);

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/home"));
  });

  test("waits for the flag before choosing", () => {
    setFlag(false, false);
    render(<ChatPage />);

    expect(screen.queryByText("chat page")).toBeNull();
    expect(routerReplace).not.toHaveBeenCalled();
  });
});
