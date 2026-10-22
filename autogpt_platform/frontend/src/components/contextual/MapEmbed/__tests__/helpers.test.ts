import { describe, expect, it } from "vitest";
import { mapLanguageSource, parseMapSource } from "../helpers";

describe("parseMapSource", () => {
  it("parses pins and a title", () => {
    const data = parseMapSource(
      JSON.stringify({
        title: "  Cafes ",
        pins: [{ name: "A", lat: 1.5, lng: 2.5, address: "1 Main" }],
      }),
    );
    expect(data?.title).toBe("Cafes");
    expect(data?.pins).toEqual([
      { name: "A", lat: 1.5, lng: 2.5, address: "1 Main" },
    ]);
  });

  it("drops pins with bad coordinates and rejects an empty map", () => {
    expect(
      parseMapSource(JSON.stringify({ pins: [{ lat: 95, lng: 0 }] })),
    ).toBeNull();
    expect(
      parseMapSource(
        JSON.stringify({
          pins: [
            { lat: "1", lng: 0 },
            { lat: 1, lng: 2 },
          ],
        }),
      )?.pins,
    ).toHaveLength(1);
  });

  it("returns null for partial JSON while a reply streams", () => {
    expect(parseMapSource('{"pins": [{"name": "A", "lat"')).toBeNull();
  });

  it("caps the number of pins", () => {
    const pins = Array.from({ length: 80 }, (_, i) => ({ lat: 1, lng: i }));
    expect(parseMapSource(JSON.stringify({ pins }))?.pins).toHaveLength(50);
  });
});

describe("mapLanguageSource", () => {
  it("reads the source of a map fence only", () => {
    expect(
      mapLanguageSource({
        props: { className: "language-map", children: "x" },
      }),
    ).toBe("x");
    expect(
      mapLanguageSource({ props: { className: "language-js", children: "x" } }),
    ).toBeNull();
    expect(mapLanguageSource("text")).toBeNull();
  });
});
