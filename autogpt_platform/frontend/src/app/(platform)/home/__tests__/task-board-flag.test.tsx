import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { Flag } from "@/services/feature-flags/use-get-flag";
import ChatPage from "../../chat/page";
import HomePage from "../page";

const { flagStatusMock, routerReplace } = vi.hoisted(() => ({
  flagStatusMock: vi.fn(() => ({
    enabled: false,
    ready: true,
    answered: true,
  })),
  routerReplace: vi.fn(),
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
    usePathname: () => "/chat",
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

  test("renders nothing until the flag resolves", () => {
    setFlag(false, false);
    const { container } = render(<HomePage />);

    expect(container.textContent).toBe("");
  });
});

describe("/chat with the task board flag", () => {
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
