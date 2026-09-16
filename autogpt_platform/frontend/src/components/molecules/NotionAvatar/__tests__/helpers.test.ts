import { describe, expect, it } from "vitest";
import {
  clampPart,
  colorForToken,
  NOTION_COLORS,
  OPTIONAL_CATEGORIES,
  PICKABLE_CATEGORIES,
  decodeNotionConfig,
  encodeNotionConfig,
  expertNotionConfig,
  isLegacyAvatarUrl,
  notionAvatarUrlFor,
  notionAvatarImageUrlFor,
  notionConfigForLegacyUrl,
  notionConfigForName,
  parseNotionAvatarUrl,
  randomNotionConfig,
  seededRandom,
  hashSeed,
} from "../helpers";
import { NOTION_CATEGORIES, NOTION_PART_COUNTS } from "../metadata.generated";

const COLOR_FAMILIES = NOTION_COLORS.map((color) => color.id);

describe("config encoding", () => {
  it("round-trips a config through its URL", () => {
    const config = notionConfigForName("Maria");

    expect(parseNotionAvatarUrl(notionAvatarUrlFor(config))).toEqual(config);
  });

  it("adds a render version to the image URL without changing stored URLs", () => {
    const config = notionConfigForName("Maria");

    expect(notionAvatarImageUrlFor(config)).toBe(
      `${notionAvatarUrlFor(config)}?v=2`,
    );
  });

  it("writes the slots in draw order", () => {
    const config = notionConfigForName("Maria");
    const slots = encodeNotionConfig(config).split(".")[0].split("-");

    expect(slots).toEqual(
      NOTION_CATEGORIES.map((category) => String(config.parts[category])),
    );
  });

  it.each([
    ["too few slots", "1-2-3.sky"],
    ["a non-numeric slot", "a-2-3-4-5-6-7-8-9-10.sky"],
    ["an unknown colour", "0-0-0-0-0-0-0-0-0-0.taupe"],
  ])("rejects %s", (_label, value) => {
    expect(decodeNotionConfig(value)).toBeNull();
  });

  it("rejects a URL that is not an avatar", () => {
    expect(parseNotionAvatarUrl("/experts/maria.svg")).toBeNull();
    expect(parseNotionAvatarUrl(null)).toBeNull();
  });
});

describe("what the picker has to offer", () => {
  it("gives a row to every feature the seeder can switch on", () => {
    // Without this, a seeded beard or set of blush marks lands on a face with
    // no control to clear it.
    for (const category of OPTIONAL_CATEGORIES) {
      expect(PICKABLE_CATEGORIES).toContain(category);
    }
  });

  it("can clear any optional feature by stepping back to nothing", () => {
    for (const category of OPTIONAL_CATEGORIES) {
      expect(clampPart(category, 1 - 1)).toBe(0);
    }
  });

  it("offers no row for a feature that is always drawn and has no blank", () => {
    expect(PICKABLE_CATEGORIES).not.toContain("face");
    expect(PICKABLE_CATEGORIES).not.toContain("nose");
  });
});

describe("part clamping", () => {
  it("wraps rather than dropping out of range, so a stale URL still draws", () => {
    expect(clampPart("hair", NOTION_PART_COUNTS.hair)).toBe(0);
    expect(clampPart("hair", NOTION_PART_COUNTS.hair + 3)).toBe(3);
    expect(clampPart("hair", -1)).toBe(NOTION_PART_COUNTS.hair - 1);
  });

  it("falls back to the first part for values that are not numbers", () => {
    expect(clampPart("hair", Number.NaN)).toBe(0);
    expect(clampPart("hair", Number.POSITIVE_INFINITY)).toBe(0);
  });
});

describe("seeding", () => {
  it("gives the same name the same face every time", () => {
    expect(notionConfigForName("Maria")).toEqual(notionConfigForName("Maria"));
    expect(notionConfigForName("Maria")).toEqual(notionConfigForName("maria"));
  });

  it("gives different names different faces", () => {
    expect(notionConfigForName("Maria")).not.toEqual(
      notionConfigForName("Frankie"),
    );
  });

  it("only ever picks parts that exist", () => {
    for (const name of ["", "a", "Zoë", "a considerably longer expert name"]) {
      const { parts } = notionConfigForName(name);
      for (const category of NOTION_CATEGORIES) {
        expect(parts[category]).toBeGreaterThanOrEqual(0);
        expect(parts[category]).toBeLessThan(NOTION_PART_COUNTS[category]);
      }
    }
  });

  it("leaves most faces without glasses or a beard", () => {
    const random = seededRandom(hashSeed("sample"));
    const faces = Array.from({ length: 400 }, () => randomNotionConfig(random));
    const bare = (category: "glasses" | "beard") =>
      faces.filter((face) => face.parts[category] === 0).length / faces.length;

    expect(bare("glasses")).toBeGreaterThan(0.4);
    expect(bare("beard")).toBeGreaterThan(0.45);
  });

  it("does not add face marks to generated avatars", () => {
    const random = seededRandom(hashSeed("sample"));
    const faces = Array.from({ length: 400 }, () => randomNotionConfig(random));

    expect(faces.every((face) => face.parts.details === 0)).toBe(true);
  });

  it("hashes a name the same way regardless of characters outside the BMP", () => {
    // "🤖" is a surrogate pair in UTF-16; hashing by code point keeps this
    // in sync with the backend's ord()-based _hash_seed.
    expect(notionAvatarUrlFor(notionConfigForName("Otto 🤖"))).toBe(
      "/avatars/notion/7-10-18-11-6-5-37-3-0-1.teal.svg",
    );
  });
});

describe("what to draw for an expert", () => {
  it("decodes an avatar the expert already has", () => {
    const config = notionConfigForName("Maria");
    const drawn = expertNotionConfig({
      name: "Maria",
      avatarUrl: notionAvatarUrlFor(config),
    });

    expect(drawn).toEqual(config);
  });

  it("seeds from the name when there is no avatar yet", () => {
    expect(expertNotionConfig({ name: "Maria", avatarUrl: null })).toEqual(
      notionConfigForName("Maria"),
    );
  });

  it("takes the accent colour the expert was given", () => {
    const drawn = expertNotionConfig({
      name: "Maria",
      avatarUrl: null,
      color: "violet-300",
    });

    expect(drawn?.color).toBe("violet");
  });

  it("returns nothing for a real picture, so the caller shows the image", () => {
    expect(
      expertNotionConfig({ name: "Maria", avatarUrl: "/experts/maria.svg" }),
    ).toBeNull();
    expect(
      expertNotionConfig({
        name: "Maria",
        avatarUrl: "https://cdn.example.com/maria.png",
      }),
    ).toBeNull();
  });

  it("resolves a legacy avatar from the old URL, not the name", () => {
    const url = "/avatars/round.sky.glasses.svg";

    expect(isLegacyAvatarUrl(url)).toBe(true);
    expect(expertNotionConfig({ name: "Maria", avatarUrl: url })).toEqual(
      notionConfigForLegacyUrl(url),
    );
    // Two experts sharing a name but raised with different shapes stay apart.
    expect(expertNotionConfig({ name: "Maria", avatarUrl: url })).not.toEqual(
      expertNotionConfig({
        name: "Maria",
        avatarUrl: "/avatars/bean.sky.crown.svg",
      }),
    );
  });

  it("keeps the colour a legacy avatar was raised with", () => {
    expect(
      expertNotionConfig({
        name: "Maria",
        avatarUrl: "/avatars/round.mint.glasses.svg",
      })?.color,
    ).toBe("emerald");
  });
});

describe("colour tokens", () => {
  it("maps a token family onto an avatar colour", () => {
    expect(colorForToken("violet-300")).toBe("violet");
    expect(colorForToken("sky-500")).toBe("sky");
  });

  it("gives every accent its own disc, so no two swatches look the same", () => {
    const discs = COLOR_FAMILIES.map((family) =>
      colorForToken(`${family}-300`),
    );

    expect(new Set(discs).size).toBe(COLOR_FAMILIES.length);
    expect(discs.every(Boolean)).toBe(true);
  });

  it("keeps every disc close to white", () => {
    for (const { disc } of NOTION_COLORS) {
      const channels = disc
        .match(/[\da-f]{2}/gi)
        ?.map((value) => Number.parseInt(value, 16));

      expect(channels).toHaveLength(3);
      const mean = (channels ?? []).reduce((sum, value) => sum + value, 0) / 3;
      expect(mean).toBeGreaterThanOrEqual(244);
    }
  });

  it("has no opinion about a family it does not know", () => {
    expect(colorForToken("chartreuse-300")).toBeNull();
    expect(colorForToken(null)).toBeNull();
  });
});
