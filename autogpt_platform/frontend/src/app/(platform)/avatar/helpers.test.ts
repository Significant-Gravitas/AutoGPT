import {
  configForName,
  decodeConfig,
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
    expect(decodeConfig("tall.neon.glasses")).toEqual({
      shape: "tall",
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
      exportFilename({ shape: "dome", color: "sky", accessory: "cap" }, "png"),
    ).toBe("expert-avatar-dome-sky-cap.png");
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
