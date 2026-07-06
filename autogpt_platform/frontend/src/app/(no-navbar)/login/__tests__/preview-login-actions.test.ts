import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  isPreviewLoginConfigured,
  loginAsPreviewAccount,
} from "../components/PreviewLoginButtons/actions";
import {
  PREVIEW_ROLES,
  PreviewRole,
} from "../components/PreviewLoginButtons/helpers";

const mockGetPreviewStealingDev = vi.fn();
vi.mock("@/services/environment", () => ({
  environment: {
    getPreviewStealingDev: () => mockGetPreviewStealingDev(),
  },
}));

const mockLogin = vi.fn();
vi.mock("../actions", () => ({
  login: (email: string, password: string) => mockLogin(email, password),
}));

describe("preview login server actions", () => {
  beforeEach(() => {
    mockGetPreviewStealingDev.mockReset();
    mockLogin.mockReset();
    vi.stubEnv("PREVIEW_ACCOUNTS_PASSWORD", "test-password");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  describe("isPreviewLoginConfigured", () => {
    test("is false outside a preview environment even with a password set", async () => {
      mockGetPreviewStealingDev.mockReturnValue(null);

      await expect(isPreviewLoginConfigured()).resolves.toBe(false);
    });

    test("is false in a preview environment without a password", async () => {
      mockGetPreviewStealingDev.mockReturnValue("some-branch");
      vi.stubEnv("PREVIEW_ACCOUNTS_PASSWORD", undefined);

      await expect(isPreviewLoginConfigured()).resolves.toBe(false);
    });

    test("is true in a preview environment with a password", async () => {
      mockGetPreviewStealingDev.mockReturnValue("some-branch");

      await expect(isPreviewLoginConfigured()).resolves.toBe(true);
    });
  });

  describe("loginAsPreviewAccount", () => {
    test("is a no-op outside a preview environment even with a password set", async () => {
      mockGetPreviewStealingDev.mockReturnValue(null);

      const result = await loginAsPreviewAccount("admin");

      expect(result).toEqual({
        success: false,
        error: "Preview login is not available",
      });
      expect(mockLogin).not.toHaveBeenCalled();
    });

    test("fails without logging in when the password is not configured", async () => {
      mockGetPreviewStealingDev.mockReturnValue("some-branch");
      vi.stubEnv("PREVIEW_ACCOUNTS_PASSWORD", undefined);

      const result = await loginAsPreviewAccount("admin");

      expect(result).toEqual({
        success: false,
        error: "Preview login is not configured",
      });
      expect(mockLogin).not.toHaveBeenCalled();
    });

    test("rejects roles outside the preview role list", async () => {
      mockGetPreviewStealingDev.mockReturnValue("some-branch");

      const result = await loginAsPreviewAccount(
        "hacker" as unknown as PreviewRole,
      );

      expect(result).toEqual({
        success: false,
        error: "Unknown preview account",
      });
      expect(mockLogin).not.toHaveBeenCalled();
    });

    test("logs in with the mapped email and password for every role", async () => {
      mockGetPreviewStealingDev.mockReturnValue("some-branch");
      mockLogin.mockResolvedValue({ success: true, next: "/" });

      for (const { role } of PREVIEW_ROLES) {
        mockLogin.mockClear();

        const result = await loginAsPreviewAccount(role);

        expect(mockLogin).toHaveBeenCalledWith(
          `preview-${role}@agpt.co`,
          "test-password",
        );
        expect(result).toEqual({ success: true, next: "/" });
      }
    });
  });
});
