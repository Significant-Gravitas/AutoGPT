import { describe, expect, it } from "vitest";
import { existsInPublic, missingPublicImage } from "../public-images";

function optimizerURL(image: string) {
  return `http://localhost:3000/_next/image?url=${encodeURIComponent(image)}&w=64&q=75`;
}

describe("the route crawl's check for images public/ doesn't have", () => {
  it("finds an image public/ doesn't have", () => {
    expect(missingPublicImage(optimizerURL("/integrations/no-such.png"))).toBe(
      "/integrations/no-such.png",
    );
  });

  it("passes an image public/ has, and one outside its folders", () => {
    expect(existsInPublic("/integrations/github.png")).toBe(true);
    expect(missingPublicImage(optimizerURL("/integrations/github.png"))).toBe(
      null,
    );
    expect(missingPublicImage(optimizerURL("/avatars/someone.png"))).toBe(null);
  });

  it("leaves an image path with a malformed percent sequence to the server", () => {
    const malformed = "/integrations/%E0%A4%A.png";

    expect(existsInPublic(malformed)).toBe(false);
    expect(missingPublicImage(optimizerURL(malformed))).toBe(null);
  });
});
