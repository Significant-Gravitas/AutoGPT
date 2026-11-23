import fs from "fs";
import os from "os";
import path from "path";
import { afterEach, describe, expect, it } from "vitest";
import {
  accessOf,
  DYNAMIC_ROUTES,
  isDynamic,
  listAppRoutes,
  ROUTE_RULES,
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
    ]);
  });
});
