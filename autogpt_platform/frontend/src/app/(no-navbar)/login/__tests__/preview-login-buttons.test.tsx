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

    await waitFor(() => expect(mockIsConfigured).toHaveBeenCalled());

    expect(screen.getByText(/PREVIEW_ACCOUNTS_PASSWORD/i)).toBeTruthy();
    expect(getButton("Admin").disabled).toBe(true);
    expect(mockLoginAsPreviewAccount).not.toHaveBeenCalled();
  });
});
