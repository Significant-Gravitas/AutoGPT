import { readdirSync } from "node:fs";
import path from "node:path";
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { getServerSession } from "@/lib/auth/server/getServerSession";

// Every admin page turns away a visitor who is not an admin, on the server,
// before it returns anything to render. Each page checks for itself and never
// relies on the middleware's check, whatever paths the middleware's matcher
// covers. Calling each page shows the check runs; reading its source would
// only show that the check is mentioned.

class SentAway extends Error {
  constructor(readonly to: string) {
    super(`sent to ${to}`);
  }
}

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  redirect: (to: string) => {
    throw new SentAway(to);
  },
  notFound: () => {
    throw new SentAway("not found");
  },
}));

type Page = (props: {
  params: Promise<Record<string, string>>;
  searchParams: Promise<Record<string, string>>;
}) => unknown;

const APP_DIR = path.resolve(__dirname, "../../..");

// The files of one kind (page or route, under any name Next takes for it)
// whose URL is under /admin. Route groups and slots leave no trace in a URL.
function filesUnderAdmin(kind: "page" | "route") {
  const name = new RegExp(`^${kind}\\.(ts|tsx|js|jsx)$`);
  return readdirSync(APP_DIR, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile() && name.test(entry.name))
    .map((entry) =>
      path.relative(APP_DIR, path.join(entry.parentPath, entry.name)),
    )
    .filter((file) => {
      const [first] = file
        .split(path.sep)
        .filter(
          (segment) => !/^\(.*\)$/.test(segment) && !segment.startsWith("@"),
        );
      return first === "admin";
    })
    .sort();
}

const adminPages = filesUnderAdmin("page");

// The pages that render nothing and only send an admin on, and where to.
const REDIRECT_ONLY: Record<string, RegExp> = {
  [path.join("(platform)", "admin", "impersonation", "page.tsx")]:
    /^\/admin\/users\b/,
  [path.join("(platform)", "admin", "rate-limits", "page.tsx")]:
    /^\/admin\/users\b/,
};

function signedInAs(role: string | null) {
  const session =
    role === null
      ? null
      : {
          session: { id: "session-1", token: "token-1" },
          user: {
            id: "user-1",
            email: "visitor@example.com",
            name: "Visitor",
            role,
            createdAt: new Date(0),
          },
        };
  vi.mocked(getServerSession).mockResolvedValue(session as never);
}

const pages = new Map<string, Page>();

// Importing a page module cold takes a while; do it once, up front, rather
// than inside the first test's timeout.
beforeAll(async () => {
  for (const file of adminPages) {
    const specifier = path.join(APP_DIR, file);
    const { default: page } = (await import(/* @vite-ignore */ specifier)) as {
      default: Page;
    };
    pages.set(file, page);
  }
}, 120_000);

async function visit(file: string) {
  const page = pages.get(file);
  if (!page) throw new Error(`${file} wasn't imported`);
  return Promise.resolve()
    .then(() =>
      page({
        params: Promise.resolve({ id: "listing-1" }),
        searchParams: Promise.resolve({}),
      }),
    )
    .catch((error: unknown) => error);
}

afterEach(() => {
  vi.unstubAllEnvs();
});

beforeEach(() => {
  // The test-data page answers "not found" off a local stack, to anyone.
  vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");
  vi.stubEnv("NEXT_PUBLIC_APP_ENV", "local");
});

describe("admin pages", () => {
  it("include each redirect-only page", () => {
    expect(adminPages).toEqual(
      expect.arrayContaining(Object.keys(REDIRECT_ONLY)),
    );
  });

  it("are found, the dynamic preview page among them", () => {
    expect(adminPages).toContain(
      path.join(
        "(platform)",
        "admin",
        "marketplace",
        "preview",
        "[id]",
        "page.tsx",
      ),
    );
    expect(adminPages.length).toBeGreaterThanOrEqual(14);
  });

  it("have no route handlers beside them", () => {
    // A route handler gets no page check. One added under /admin needs its
    // own session check, and a test of it listed here.
    expect(filesUnderAdmin("route")).toEqual([]);
  });

  describe.each(adminPages)("%s", (file) => {
    it.each([
      ["a signed-out visitor", null],
      ["a signed-in member", "user"],
    ])("turns %s away without rendering", async (_visitor, role) => {
      signedInAs(role);

      const outcome = await visit(file);

      expect(
        outcome,
        'Run the check before returning: `await (await withRoleAccess(["admin"]))(Component)`',
      ).toBeInstanceOf(SentAway);
    });

    it("lets an admin in", async () => {
      signedInAs("admin");

      const outcome = await visit(file);

      const destination = REDIRECT_ONLY[file];
      if (destination) {
        expect(outcome).toBeInstanceOf(SentAway);
        expect((outcome as SentAway).to).toMatch(destination);
        return;
      }
      // Sent anywhere, Home included, an admin never sees the page (#17308).
      expect(outcome, "an admin should get the page itself").not.toBeInstanceOf(
        Error,
      );
      expect(outcome).toBeTruthy();
    });
  });
});
