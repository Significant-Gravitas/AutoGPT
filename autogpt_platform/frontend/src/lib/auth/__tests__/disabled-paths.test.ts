// @vitest-environment node
import { beforeAll, describe, expect, it, vi } from "vitest";
import { DISABLED_AUTH_PATHS } from "../disabled-paths";

vi.unmock("@/lib/auth/auth");

vi.mock("pg", () => ({
  Pool: class {
    query(): never {
      throw new Error("no database in this test");
    }
    connect(): never {
      throw new Error("no database in this test");
    }
    on() {}
    end() {}
  },
}));

type Auth = typeof import("../auth").auth;
let auth: Auth;

beforeAll(async () => {
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-".padEnd(48, "x"));
  vi.stubEnv("BETTER_AUTH_URL", "http://localhost:3000");
  ({ auth } = await import("../auth"));
}, 120_000);

// A query string doesn't change the path the router matches.
const SPELLINGS = [(path: string) => path, (path: string) => `${path}?x=1`];

// Every endpoint the platform serves on purpose. A Better Auth upgrade or a
// new plugin that mounts another one fails the test below, so each new
// endpoint is a decision: add it here if something calls it, otherwise to
// DISABLED_AUTH_PATHS.
const SERVED_AUTH_PATHS = [
  "/admin/ban-user",
  "/admin/create-user",
  "/admin/get-user",
  "/admin/list-users",
  "/admin/unban-user",
  "/callback/:id",
  "/change-email",
  "/error",
  "/get-session",
  "/jwks",
  "/ok",
  "/request-password-reset",
  "/reset-password",
  "/reset-password/:token",
  "/send-verification-email",
  "/sign-in/email",
  "/sign-in/social",
  "/sign-out",
  "/sign-up/email",
  "/supabase-bridge",
  "/token",
  "/update-user",
  "/verify-email",
];

function mountedEndpoints() {
  return Object.values(auth.api).flatMap((endpoint) => {
    const { path, options } = endpoint as unknown as {
      path?: string;
      options: { method: string | string[] };
    };
    // Server-only endpoints have no path: they can't be requested at all.
    if (!path) return [];
    const methods = [options.method].flat();
    return [{ path, methods: methods.includes("*") ? ["GET"] : methods }];
  });
}

async function statusOf(path: string, method: string) {
  const response = await auth.handler(
    new Request(`http://localhost:3000/api/auth${path}`, {
      method,
      headers: {
        "content-type": "application/json",
        origin: "http://localhost:3000",
      },
      body: method === "GET" ? undefined : "{}",
    }),
  );
  return response.status;
}

describe("Better Auth endpoints", () => {
  it("are each either served on purpose or disabled", () => {
    const mounted = [...new Set(mountedEndpoints().map(({ path }) => path))];
    const served = mounted.filter(
      (path) => !DISABLED_AUTH_PATHS.includes(path),
    );
    expect(served.sort()).toEqual([...SERVED_AUTH_PATHS].sort());
    // And every disabled path is a real endpoint, so none of them is a typo.
    expect(
      DISABLED_AUTH_PATHS.filter((path) => !mounted.includes(path)),
    ).toEqual([]);
  });

  it.each(DISABLED_AUTH_PATHS)("%s is not served", async (path) => {
    const endpoint = mountedEndpoints().find((e) => e.path === path);
    expect(endpoint, path).toBeDefined();
    for (const spell of SPELLINGS) {
      for (const method of endpoint?.methods ?? []) {
        const status = await statusOf(spell(path), method);
        expect(status, `${method} ${spell(path)}`).toBe(404);
      }
    }
  });

  it.each(SERVED_AUTH_PATHS)("%s is still served", async (path) => {
    const endpoint = mountedEndpoints().find((e) => e.path === path);
    expect(endpoint, path).toBeDefined();
    for (const method of endpoint?.methods ?? []) {
      const status = await statusOf(path.replace(/:\w+/g, "x"), method);
      expect(status, `${method} ${path}`).not.toBe(404);
    }
  });
});
