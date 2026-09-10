import { describe, expect, it } from "vitest";
import { GET } from "../route";

describe("GET /avatars/[file]", () => {
  it("renders a generated avatar as SVG", async () => {
    const response = await GET(new Request("http://localhost/avatars/x"), {
      params: Promise.resolve({ file: "round.sky.none.svg" }),
    });

    expect(response.status).toBe(200);
    expect(response.headers.get("Content-Type")).toContain("image/svg+xml");
    const body = await response.text();
    expect(body.startsWith("<svg xmlns=")).toBe(true);
  });

  it("404s on a file that is not an avatar spec", async () => {
    const response = await GET(new Request("http://localhost/avatars/x"), {
      params: Promise.resolve({ file: "nope.svg" }),
    });

    expect(response.status).toBe(404);
  });
});
