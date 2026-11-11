import {
  expertTint,
  identityFamilies,
  mixHex,
  OTTO_TINT,
  palette,
} from "../expertPalette";

interface Catalog {
  palette: Record<string, { label: string; hex: string }>;
  identities: { id: string; visual_category: string }[];
}

// The web's own copy, read by relative path (the app has no Node types for `fs`).
const catalog = jest.requireActual<Catalog>(
  "../../../../frontend/src/components/molecules/ExpertAvatar/catalog.json",
);

describe("expertPalette", () => {
  it("matches the web catalog", () => {
    expect(palette).toEqual(catalog.palette);
    expect(identityFamilies).toEqual(
      Object.fromEntries(
        catalog.identities.map((identity) => [identity.id, identity.visual_category]),
      ),
    );
  });

  it("mixes a hex over a base per channel", () => {
    // 182*.24+255*.76 = 237.48, 164*.24+255*.76 = 233.16, 200*.24+255*.76 = 241.8
    expect(mixHex("#B6A4C8", 24, "#FFFFFF")).toBe("#EDE9F2");
  });

  it("gives Otto's role the Otto tint", () => {
    expect(expertTint({ role: "Head of AI", categories: ["sales"] })).toBe(OTTO_TINT);
  });

  it("takes a managed identity's family from its URL", () => {
    expect(
      expertTint({
        avatarUrl:
          "https://platform.agpt.co/autogpt-characters/v2.1/expert-max/neutral/128.webp",
        categories: ["finance"],
      }),
    ).toBe(palette.sales.hex);
  });

  it("takes a custom avatar's first family category", () => {
    expect(
      expertTint({
        avatarUrl: "https://example.com/me.png",
        categories: ["Otto", "unknown", "Research", "sales"],
      }),
    ).toBe(palette.research.hex);
  });
});
