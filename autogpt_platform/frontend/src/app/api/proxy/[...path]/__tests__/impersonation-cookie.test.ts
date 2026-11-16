// @vitest-environment node
import { RequestCookies } from "next/dist/compiled/@edge-runtime/cookies";
import { describe, expect, it } from "vitest";
import { decodeImpersonation, encodeImpersonation } from "@/lib/impersonation";

// The browser writes the impersonation record URL-encoded (the cookie writer
// runs encodeURIComponent). The proxy reads it back through Next's request
// cookie jar, which must hand back the decoded JSON.
describe("impersonation cookie as the proxy reads it", () => {
  it("comes back decoded, so the record parses", () => {
    const record = { target: "user-a", owner: "admin-1" };
    const headers = new Headers({
      cookie: `other=1; admin-impersonate-user-id=${encodeURIComponent(
        encodeImpersonation(record),
      )}`,
    });

    const value = new RequestCookies(headers).get(
      "admin-impersonate-user-id",
    )?.value;

    expect(decodeImpersonation(value)).toEqual(record);
  });
});
