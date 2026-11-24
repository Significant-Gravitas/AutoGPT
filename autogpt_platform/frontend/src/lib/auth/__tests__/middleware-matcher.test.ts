// @vitest-environment node
import { unstable_doesMiddlewareMatch } from "next/experimental/testing/server";
import { describe, expect, it } from "vitest";
import { config } from "@/middleware";

// The admin gate runs on every admin path, including one that ends like an
// image, which the main pattern would skip.
describe("the middleware matcher", () => {
  it.each([
    "/admin/users",
    "/admin/marketplace/preview/listing-1",
    "/admin/marketplace/preview/listing-1.png",
    "/admin/settings.svg",
  ])("runs on %s", (url) => {
    expect(unstable_doesMiddlewareMatch({ config, url })).toBe(true);
  });

  it("still skips an image outside /admin", () => {
    expect(
      unstable_doesMiddlewareMatch({ config, url: "/integrations/linear.png" }),
    ).toBe(false);
  });
});
