import fs from "fs";
import os from "os";
import path from "path";
import { afterEach, describe, expect, it } from "vitest";
import {
  accessOf,
  type DynamicRoute,
  DYNAMIC_ROUTES,
  isDynamic,
  KNOWN_BUGS,
  listAppRoutes,
  ROUTE_RULES,
  samplingGaps,
} from "../route-table";

const routes = listAppRoutes();

describe("the route crawl's table", () => {
  it("finds the app's pages, by the paths Next.js serves them at", () => {
    expect(routes).toEqual(
      expect.arrayContaining([
        "/",
        "/login",
        "/settings/billing",
        "/library/agents/[id]",
        "/marketplace/agent/[creator]/[slug]",
      ]),
    );
    expect(routes.filter((route) => /[()@]/.test(route))).toEqual([]);
  });

  it("names only pages that exist", () => {
    expect(
      Object.keys(ROUTE_RULES).filter((route) => !routes.includes(route)),
    ).toEqual([]);
    expect(
      Object.keys(DYNAMIC_ROUTES).filter(
        (route) => !routes.includes(route) || !isDynamic(route),
      ),
    ).toEqual([]);
  });

  it("pins bugs only on pages that exist", () => {
    expect(
      KNOWN_BUGS.map((bug) => bug.route).filter(
        (route) => !routes.includes(route),
      ),
    ).toEqual([]);
  });

  it("sends pages on only to pages that exist", () => {
    const targets = Object.values(ROUTE_RULES).flatMap((rule) =>
      [rule.landsOn, rule.signedInLandsOn].filter(
        (target): target is string => target !== undefined,
      ),
    );
    expect(targets.filter((target) => !routes.includes(target))).toEqual([]);
  });

  it("says why each page needs an entry", () => {
    expect(
      Object.entries(ROUTE_RULES)
        .filter(([, rule]) => !rule.reason.trim())
        .map(([route]) => route),
    ).toEqual([]);
  });

  it("finds a dynamic route a test is given and has no sample for, and one it isn't given", () => {
    const table = {
      "/a/[id]": { sampledIn: "shared", sample: "an a" },
      "/b/[id]": { sampledIn: "shared", sample: "a b" },
      "/c/[id]": { sampledIn: "admin", sample: "a c" },
    } satisfies Record<string, DynamicRoute>;

    expect(samplingGaps("shared", ["/a/[id]", "/b/[id]"], table)).toEqual({
      missing: [],
      misplaced: [],
    });
    expect(samplingGaps("shared", ["/a/[id]"], table)).toEqual({
      missing: ["/b/[id]"],
      misplaced: [],
    });
    expect(
      samplingGaps(
        "shared",
        ["/a/[id]", "/b/[id]", "/c/[id]", "/d/[id]"],
        table,
      ),
    ).toEqual({ missing: [], misplaced: ["/c/[id]", "/d/[id]"] });
  });

  it("says what each dynamic route is opened with", () => {
    expect(
      Object.entries(DYNAMIC_ROUTES)
        .filter(([, { sample }]) => !sample.trim())
        .map(([route]) => route),
    ).toEqual([]);
  });

  it("crawls a page that sends the user on only where the page it sends them to is served", () => {
    const unmarked = Object.entries(ROUTE_RULES).filter(([, rule]) =>
      [rule.landsOn, rule.signedInLandsOn].some((target) => {
        const destination = target ? ROUTE_RULES[target] : undefined;
        return (
          (destination?.hostedOnly && !rule.hostedOnly) ||
          (destination?.localOnly && !rule.localOnly)
        );
      }),
    );
    expect(unmarked.map(([route]) => route)).toEqual([]);
  });

  it("crawls /admin pages as an admin and the rest signed in by default", () => {
    expect(accessOf("/admin/users")).toBe("admin");
    expect(accessOf("/library")).toBe("signed-in");
    expect(accessOf("/login")).toBe("public");
  });
});

describe("listAppRoutes", () => {
  let appDir: string | undefined;

  afterEach(() => {
    if (appDir) fs.rmSync(appDir, { recursive: true, force: true });
  });

  it("drops route groups and slots, and skips private folders and intercepting routes", () => {
    appDir = fs.mkdtempSync(path.join(os.tmpdir(), "route-table-"));
    for (const page of [
      "page.tsx",
      "(platform)/library/page.tsx",
      "(platform)/library/agents/[id]/page.tsx",
      "(group)/(nested)/docs/[[...slug]]/page.mdx",
      "dashboard/@modal/settings/page.tsx",
      "reports/page.tsx",
      "reports/@chart/page.tsx",
      "_private/page.tsx",
      "feed/(.)photo/[id]/page.tsx",
      "feed/components/Card.tsx",
    ]) {
      fs.mkdirSync(path.dirname(path.join(appDir, page)), { recursive: true });
      fs.writeFileSync(path.join(appDir, page), "");
    }

    expect(listAppRoutes(appDir)).toEqual([
      "/",
      "/dashboard/settings",
      "/docs/[[...slug]]",
      "/library",
      "/library/agents/[id]",
      "/reports",
    ]);
  });
});
