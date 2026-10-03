import { describe, expect, it } from "vitest";
import { GET } from "../route";

function get(file: string) {
  return GET(new Request("http://localhost/avatars/notion/x"), {
    params: Promise.resolve({ file }),
  });
}

const VALID = "2-3-5-4-2-0-12-0-0-0.sky.svg";

describe("GET /avatars/notion/[file]", () => {
  it("renders a composed avatar as immutable SVG", async () => {
    const response = await get(VALID);

    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toContain("image/svg+xml");
    expect(response.headers.get("Cache-Control")).toContain("immutable");
    const body = await response.text();
    expect(body.startsWith("<svg xmlns=")).toBe(true);
    expect(body).toContain("scale(1.2)");
    expect(body).toContain("</svg>");
  });

  it("stacks every layer in draw order", async () => {
    const body = await (await get(VALID)).text();

    expect(body.match(/<g>/g)).toHaveLength(10);
    // No layer paints the skin: it takes the disc's colour, so the face sits
    // in its circle rather than on a white cut-out.
    expect(body).not.toContain('fill="#ffffff"');
  });

  it("clips itself to the disc, since the head overflows the artboard", async () => {
    const body = await (await get(VALID)).text();

    expect(body).toContain("<clipPath");
    expect(body).toMatch(/<g clip-path="url\(#[^)]+\)">/);
  });

  it("gives each request's ids a prefix so two avatars can share a page", async () => {
    const body = await (await get(VALID)).text();

    expect(body).not.toContain("{{P}}");
  });

  it("wraps an out-of-range index instead of failing", async () => {
    const wrapped = await (await get("0-0-0-0-0-0-9999-0-0-0.sky.svg")).text();
    const expected = await (
      await get(`0-0-0-0-0-0-${9999 % 59}-0-0-0.sky.svg`)
    ).text();

    expect(wrapped).toBe(expected);
  });

  it.each([
    ["a slug that is not ten slots", "1-2-3.sky.svg"],
    ["a non-numeric slot", "a-2-3-4-5-6-7-8-9-10.sky.svg"],
    ["an unknown colour", "0-0-0-0-0-0-0-0-0-0.taupe.svg"],
    ["a missing extension", "0-0-0-0-0-0-0-0-0-0.sky"],
  ])("404s on %s", async (_label, file) => {
    expect((await get(file)).status).toBe(404);
  });
});
