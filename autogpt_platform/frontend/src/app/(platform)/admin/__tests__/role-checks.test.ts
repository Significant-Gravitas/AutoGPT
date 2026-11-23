// @vitest-environment node
import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

// Every admin page checks the visitor's role on the server, from their
// session, as well as the middleware's check before it.
const ADMIN_DIR = path.resolve(__dirname, "..");

// Pages that render nothing and only redirect to another admin page, which
// checks the role itself.
const REDIRECT_ONLY = ["impersonation/page.tsx", "rate-limits/page.tsx"];

function adminPages() {
  return readdirSync(ADMIN_DIR, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name === "page.tsx")
    .map((entry) =>
      path.relative(ADMIN_DIR, path.join(entry.parentPath, entry.name)),
    )
    .sort();
}

// A page's code without its comments, so a mention in one doesn't count.
function source(page: string) {
  return readFileSync(path.join(ADMIN_DIR, page), "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/.*$/gm, "$1");
}

describe("admin pages", () => {
  it("include the preview page and the redirect-only pages", () => {
    expect(adminPages()).toEqual(
      expect.arrayContaining([
        "marketplace/preview/[id]/page.tsx",
        ...REDIRECT_ONLY,
      ]),
    );
  });

  it.each(adminPages().filter((page) => !REDIRECT_ONLY.includes(page)))(
    "%s checks the admin role on the server",
    (page) => {
      expect(
        source(page),
        'Wrap the page in withRoleAccess(["admin"]), as admin/users/page.tsx does',
      ).toContain('withRoleAccess(["admin"])');
    },
  );

  it.each(REDIRECT_ONLY)("%s renders nothing, only redirects", (page) => {
    const code = source(page);
    expect(code).toContain("redirect(getAdminUsersHref(");
    expect(
      code,
      'A page that renders needs withRoleAccess(["admin"]): take it off REDIRECT_ONLY',
    ).not.toMatch(/\breturn\b/);
  });
});
