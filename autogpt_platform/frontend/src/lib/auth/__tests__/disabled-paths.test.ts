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

// Spellings the URL parser turns into the plain path, so the router would
// match them.
const SPELLINGS = [
  (path: string) => path,
  (path: string) => path.replace(/\/([^/]+)$/, "/./$1"),
  (path: string) => path.replace(/\/([^/]+)$/, "\\$1"),
  (path: string) => `${path}?x=1`,
];

async function statusOf(path: string, method: string) {
  const response = await auth.handler(
    new Request(`http://localhost:3000/api/auth${path}`, {
      method,
      headers: {
        "content-type": "application/json",
        origin: "http://localhost:3000",
      },
      body: method === "POST" ? "{}" : undefined,
    }),
  );
  return response.status;
}

describe("Better Auth endpoints nothing calls", () => {
  it("are each a real endpoint, so none of them is a typo", () => {
    const mounted = new Set(
      Object.values(auth.api).map((endpoint) => endpoint.path),
    );
    expect(DISABLED_AUTH_PATHS.filter((path) => !mounted.has(path))).toEqual(
      [],
    );
  });

  it.each(DISABLED_AUTH_PATHS)("%s is not served", async (path) => {
    for (const spell of SPELLINGS) {
      for (const method of ["GET", "POST"]) {
        expect(await statusOf(spell(path), method), spell(path)).toBe(404);
      }
    }
  });

  it.each([
    ["/get-session", "GET"],
    ["/sign-in/email", "POST"],
    ["/admin/list-users", "GET"],
    ["/admin/get-user", "GET"],
    ["/admin/ban-user", "POST"],
    ["/admin/unban-user", "POST"],
    ["/admin/create-user", "POST"],
  ])("%s, which the platform uses, is still served", async (path, method) => {
    expect(await statusOf(path, method)).not.toBe(404);
  });
});
