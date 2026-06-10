import { vi } from "vitest";

export function mockAuthRequest() {
  // Keep the Better Auth server instance (pg pool, plugins) and next/headers
  // out of the vitest module graph. The defaults emulate "no active session";
  // tests that need user data should mock @/lib/auth/hooks/useAuth or
  // @/lib/auth/actions in their own file.
  vi.mock("@/lib/auth/auth", () => ({
    auth: {
      api: {
        getSession: vi.fn().mockResolvedValue(null),
        signOut: vi.fn().mockResolvedValue({ success: true }),
        getToken: vi.fn().mockResolvedValue(null),
      },
    },
  }));
  vi.mock("@/lib/auth/server/getServerSession", () => ({
    getServerSession: vi.fn().mockResolvedValue(null),
  }));
  vi.mock("@/lib/auth/server/token", () => ({
    getBackendAuthToken: vi.fn().mockResolvedValue(null),
  }));
}
