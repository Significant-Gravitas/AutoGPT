import {
  avatarUrlFor,
  colorForToken,
  configForName,
  decodeConfig,
  expertAvatarConfig,
  isUploadedAvatar,
  parseAvatarUrl,
  DEFAULT_CONFIG,
  encodeConfig,
  randomConfig,
  seededRandom,
} from "@/components/molecules/BotAvatar/helpers";
import { describe, expect, test } from "vitest";
import { exportFilename, rosterConfigs, serializeSvg } from "./helpers";

describe("avatar config codec", () => {
  test("round-trips a config through the query string", () => {
    const config = {
      shape: "bean",
      color: "mint",
      accessory: "headset",
    } as const;
    expect(decodeConfig(encodeConfig(config))).toEqual(config);
  });

  test("falls back per field on junk input", () => {
    expect(decodeConfig(null)).toEqual(DEFAULT_CONFIG);
    expect(decodeConfig("dome.neon.glasses")).toEqual({
      shape: "dome",
      color: DEFAULT_CONFIG.color,
      accessory: "glasses",
    });
    expect(decodeConfig("nonsense")).toEqual(DEFAULT_CONFIG);
  });

  test("random config only ever emits known ids", () => {
    const random = seededRandom(42);
    for (let index = 0; index < 50; index += 1) {
      const config = randomConfig(random);
      expect(decodeConfig(encodeConfig(config))).toEqual(config);
    }
  });

  test("name-seeded config is stable and differs across names", () => {
    expect(configForName("Otto")).toEqual(configForName("otto"));
    const distinct = new Set(
      rosterConfigs().map((member) => encodeConfig(member.config)),
    );
    expect(distinct.size).toBeGreaterThan(1);
  });
});

describe("export helpers", () => {
  test("filename encodes the config", () => {
    expect(
      exportFilename({ shape: "dome", color: "sky", accessory: "bow" }, "png"),
    ).toBe("expert-avatar-dome-sky-bow.png");
  });

  test("serialized svg is standalone and export sized", () => {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "shrink-0");
    svg.setAttribute("viewBox", "0 0 120 120");
    const markup = serializeSvg(svg);
    expect(markup).toContain('xmlns="http://www.w3.org/2000/svg"');
    expect(markup).toContain('width="512"');
    expect(markup).not.toContain("class=");
  });
});

describe("expert avatar resolution", () => {
  test("a generated avatar url round-trips and an upload does not parse", () => {
    const config = { shape: "wide", color: "coral", accessory: "bow" } as const;
    expect(parseAvatarUrl(avatarUrlFor(config))).toEqual(config);
    expect(parseAvatarUrl("https://cdn.example/otto.png")).toBeNull();
    expect(parseAvatarUrl("/avatars/cube.neon.hat.svg")).toBeNull();
    expect(isUploadedAvatar("https://cdn.example/otto.png")).toBe(true);
    expect(isUploadedAvatar(avatarUrlFor(config))).toBe(false);
    expect(isUploadedAvatar(null)).toBe(false);
  });

  test("an expert without an upload is seeded from its name and accent", () => {
    const seeded = expertAvatarConfig({
      name: "Otto",
      avatarUrl: null,
      color: "sky-300",
    });
    expect(seeded).toEqual({ ...configForName("Otto"), color: "sky" });
    expect(
      expertAvatarConfig({ name: "Otto", avatarUrl: null, color: "" }),
    ).toEqual(configForName("Otto"));
    expect(colorForToken("fuchsia-300")).toBe("plum");
    expect(colorForToken("neon-300")).toBeNull();
  });
});
