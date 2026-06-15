import { render, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ErrorPage from "../page";

let searchMessage: string | null = null;
let supabaseState = {
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

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => supabaseState,
}));

describe("ErrorPage", () => {
  afterEach(() => {
    searchMessage = null;
    supabaseState = {
      isUserLoading: false,
      isLoggedIn: false,
    };
    replaceMock.mockClear();
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
    supabaseState = {
      isUserLoading: false,
      isLoggedIn: true,
    };

    const { container } = render(<ErrorPage />);

    await waitFor(() => {
      expect(container.querySelector(".min-h-screen")).not.toBeNull();
    });

    expect(replaceMock).not.toHaveBeenCalled();
  });
});
