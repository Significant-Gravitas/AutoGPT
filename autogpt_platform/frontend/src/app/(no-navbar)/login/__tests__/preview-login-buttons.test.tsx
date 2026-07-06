import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { PreviewLoginButtons } from "../components/PreviewLoginButtons/PreviewLoginButtons";

const mockGetPreviewStealingDev = vi.fn();
vi.mock("@/services/environment", () => ({
  environment: {
    getPreviewStealingDev: () => mockGetPreviewStealingDev(),
  },
}));

const mockIsConfigured = vi.fn();
const mockLoginAsPreviewAccount = vi.fn();
vi.mock("../components/PreviewLoginButtons/actions", () => ({
  isPreviewLoginConfigured: () => mockIsConfigured(),
  loginAsPreviewAccount: (role: string) => mockLoginAsPreviewAccount(role),
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
}));

const toastSpy = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: toastSpy }),
}));

const ROLE_LABELS = [
  "Admin",
  "Existing user",
  "Clean user",
  "Pro",
  "Enterprise",
];

function getButton(label: string) {
  return screen.getByRole("button", { name: label }) as HTMLButtonElement;
}

describe("PreviewLoginButtons", () => {
  beforeEach(() => {
    mockGetPreviewStealingDev.mockReset();
    mockIsConfigured.mockReset();
    mockLoginAsPreviewAccount.mockReset();
    toastSpy.mockClear();
  });

  test("renders nothing outside a preview environment", () => {
    mockGetPreviewStealingDev.mockReturnValue(null);
    mockIsConfigured.mockResolvedValue(true);

    render(<PreviewLoginButtons />);

    expect(screen.queryByText(/Preview test accounts/i)).toBeNull();
    expect(screen.queryByRole("button", { name: "Admin" })).toBeNull();
  });

  test("shows all role buttons enabled in a configured preview environment", async () => {
    mockGetPreviewStealingDev.mockReturnValue("some-branch");
    mockIsConfigured.mockResolvedValue(true);

    render(<PreviewLoginButtons />);

    expect(screen.getByText(/Preview test accounts/i)).toBeTruthy();
    for (const label of ROLE_LABELS) {
      expect(getButton(label)).toBeTruthy();
    }

    await waitFor(() => expect(getButton("Admin").disabled).toBe(false));
    expect(screen.queryByText(/PREVIEW_ACCOUNTS_PASSWORD/i)).toBeNull();
  });

  test("clicking a role button logs in as that role", async () => {
    mockGetPreviewStealingDev.mockReturnValue("some-branch");
    mockIsConfigured.mockResolvedValue(true);
    // Never resolves so the handler does not attempt a navigation in jsdom.
    mockLoginAsPreviewAccount.mockReturnValue(new Promise(() => {}));

    render(<PreviewLoginButtons />);

    await waitFor(() => expect(getButton("Pro").disabled).toBe(false));

    fireEvent.click(getButton("Pro"));

    expect(mockLoginAsPreviewAccount).toHaveBeenCalledWith("pro");
  });

  test("renders a disabled state with a hint when the password is not configured", async () => {
    mockGetPreviewStealingDev.mockReturnValue("some-branch");
    mockIsConfigured.mockResolvedValue(false);

    render(<PreviewLoginButtons />);

    expect(await screen.findByText(/PREVIEW_ACCOUNTS_PASSWORD/i)).toBeTruthy();
    expect(getButton("Admin").disabled).toBe(true);
    expect(mockLoginAsPreviewAccount).not.toHaveBeenCalled();
  });

  test("does not show the setup hint while the config check is loading", () => {
    mockGetPreviewStealingDev.mockReturnValue("some-branch");
    // Never resolves so the component stays in its initial checking state.
    mockIsConfigured.mockReturnValue(new Promise(() => {}));

    render(<PreviewLoginButtons />);

    expect(screen.getByText(/Preview test accounts/i)).toBeTruthy();
    expect(screen.queryByText(/PREVIEW_ACCOUNTS_PASSWORD/i)).toBeNull();
    expect(getButton("Admin").disabled).toBe(true);
  });

  test("shows a toast when the preview login fails", async () => {
    mockGetPreviewStealingDev.mockReturnValue("some-branch");
    mockIsConfigured.mockResolvedValue(true);
    mockLoginAsPreviewAccount.mockResolvedValue({
      success: false,
      error: "Preview login is not available",
    });

    render(<PreviewLoginButtons />);

    await waitFor(() => expect(getButton("Admin").disabled).toBe(false));

    fireEvent.click(getButton("Admin"));

    await waitFor(() =>
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Preview login is not available",
          variant: "destructive",
        }),
      ),
    );
  });
});
