import { describe, expect, it } from "vitest";
import {
  ACCESSORIES,
  avatarUrlFor,
  configForName,
  isAccessoryId,
  parseAvatarUrl,
} from "../helpers";

describe("avatar urls", () => {
  it.each(ACCESSORIES.map((option) => option.id))(
    "round-trips the %s accessory through the url",
    (accessory) => {
      const config = { shape: "round", color: "sky", accessory } as const;
      expect(parseAvatarUrl(avatarUrlFor(config))).toEqual(config);
    },
  );

  it("still parses the original accessory ids", () => {
    const legacy = [
      "glasses",
      "headset",
      "star",
      "bow",
      "badge",
      "crown",
      "propeller",
      "ears",
      "flower",
      "bowtie",
      "headband",
      "none",
    ];
    for (const id of legacy) {
      expect(isAccessoryId(id)).toBe(true);
      expect(parseAvatarUrl(`/avatars/dome.plum.${id}.svg`)).toEqual({
        shape: "dome",
        color: "plum",
        accessory: id,
      });
    }
  });

  it("rejects unknown accessories", () => {
    expect(isAccessoryId("sombrero")).toBe(false);
    expect(parseAvatarUrl("/avatars/round.sky.sombrero.svg")).toBeNull();
  });

  it("seeds the same config for the same name", () => {
    expect(configForName("Otto")).toEqual(configForName("otto"));
  });
});
