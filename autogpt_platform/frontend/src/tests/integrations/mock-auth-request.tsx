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
  vi.mock("@/lib/auth/server/getServerSession", async () => {
    const { isAPIError } = await import("better-auth/api");
    return {
      getServerSession: vi.fn().mockResolvedValue(null),
      readServerSession: vi.fn().mockResolvedValue(null),
      isSessionRejection: vi.fn(
        (error: unknown) => isAPIError(error) && error.statusCode === 401,
      ),
    };
  });
}
