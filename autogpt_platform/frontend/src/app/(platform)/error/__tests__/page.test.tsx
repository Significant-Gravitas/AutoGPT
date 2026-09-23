import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ErrorPage from "../page";

let searchMessage: string | null = null;
let authState = {
  isUserLoading: false,
  isLoggedIn: false,
};
const replaceMock = vi.fn();

vi.mock("next/navigation", () => ({
  useSearchParams: () => ({
    get: (key: string) => (key === "message" ? searchMessage : null),
  }),
  useRouter: () => ({
    replace: replaceMock,
  }),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => authState,
}));

describe("ErrorPage", () => {
  afterEach(() => {
    searchMessage = null;
    authState = {
      isUserLoading: false,
      isLoggedIn: false,
    };
    replaceMock.mockClear();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("redirects logged-out users away from the session expired screen", async () => {
    searchMessage = "session-expired";

    render(<ErrorPage />);

    await waitFor(() => {
      expect(replaceMock).toHaveBeenCalledWith("/login");
    });
  });

  it("keeps the session expired screen for authenticated users", async () => {
    searchMessage = "session-expired";
    authState = {
      isUserLoading: false,
      isLoggedIn: true,
    };

    render(<ErrorPage />);

    await waitFor(() => {
      expect(screen.queryByText(/Your session has expired/)).not.toBeNull();
    });

    expect(replaceMock).not.toHaveBeenCalled();
  });

  it("waits for authentication before showing an expired session or redirecting", () => {
    searchMessage = "session-expired";
    authState = { isUserLoading: true, isLoggedIn: false };

    const { rerender } = render(<ErrorPage />);
    expect(screen.queryByText(/Your session has expired/)).toBeNull();
    expect(replaceMock).not.toHaveBeenCalled();

    authState = { isUserLoading: false, isLoggedIn: false };
    rerender(<ErrorPage />);
    expect(replaceMock).toHaveBeenCalledWith("/login");
  });

  it("preserves unrelated errors for logged-out users", () => {
    searchMessage = "server-error";
    render(<ErrorPage />);
    expect(
      screen.queryByText(/Our servers are experiencing issues/),
    ).not.toBeNull();
    expect(replaceMock).not.toHaveBeenCalled();
  });

  it.each([
    "session-expired",
    "auth-failed",
    "auth-token-invalid",
    "user-creation-failed",
  ])(
    "restarts authentication for %s when cached auth still says logged in",
    (message) => {
      const navigate = vi
        .spyOn(window.location, "replace")
        .mockImplementation(() => {});
      searchMessage = message;
      authState = { isUserLoading: false, isLoggedIn: true };
      render(<ErrorPage />);

      fireEvent.click(screen.getByRole("button", { name: "Try Again" }));

      expect(navigate).toHaveBeenCalledExactlyOnceWith("/login");
      expect(replaceMock).not.toHaveBeenCalled();
    },
  );

  it("returns home when retrying a general server error", () => {
    searchMessage = "server-error";
    render(<ErrorPage />);

    fireEvent.click(screen.getByRole("button", { name: "Try Again" }));

    expect(replaceMock).toHaveBeenCalledExactlyOnceWith("/");
  });

  it("waits before retrying a rate-limited request", () => {
    vi.useFakeTimers();
    const reload = vi
      .spyOn(window.location, "reload")
      .mockImplementation(() => {});
    searchMessage = "rate-limited";
    render(<ErrorPage />);

    fireEvent.click(screen.getByRole("button", { name: "Try Again" }));
    expect(reload).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(2000));

    expect(reload).toHaveBeenCalledTimes(1);
    expect(replaceMock).not.toHaveBeenCalled();
  });
});
