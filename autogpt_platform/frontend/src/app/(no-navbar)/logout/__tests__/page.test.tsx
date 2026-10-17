import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockLogOut = vi.hoisted(() => vi.fn());
const mockToast = vi.hoisted(() => vi.fn());
const mockReplace = vi.hoisted(() => vi.fn());

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ logOut: mockLogOut }),
}));
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  useToast: () => ({ toast: mockToast, dismiss: vi.fn() }),
}));
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mockReplace, push: vi.fn(), refresh: vi.fn() }),
}));

import LogoutPage from "../page";

beforeEach(() => {
  mockLogOut.mockReset();
  mockToast.mockReset();
  mockReplace.mockReset();
});

describe("LogoutPage", () => {
  test("goes to the login page once the sign-out has gone through", async () => {
    mockLogOut.mockResolvedValue(undefined);

    render(<LogoutPage />);

    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith("/login"));
    expect(mockLogOut).toHaveBeenCalledTimes(1);
    expect(mockToast).not.toHaveBeenCalled();
  });

  test("says the sign-out failed and stays in the app when it never reached the server", async () => {
    mockLogOut.mockRejectedValue(new TypeError("Failed to fetch"));

    render(<LogoutPage />);

    await waitFor(() => expect(mockReplace).toHaveBeenCalledWith("/"));
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Couldn't sign you out",
        variant: "destructive",
      }),
    );
    expect(mockReplace).not.toHaveBeenCalledWith("/login");
  });
});
