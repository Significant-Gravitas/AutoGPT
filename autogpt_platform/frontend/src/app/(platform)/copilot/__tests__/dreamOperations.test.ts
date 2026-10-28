import type { UIMessage } from "ai";
import { describe, expect, it, vi } from "vitest";

import {
  DREAM_OPERATIONS_PART_TYPE,
  handleDreamOperationsPart,
  isDreamOperationsPart,
  readDreamOperationsPart,
} from "../dreamOperations";

type Part = UIMessage["parts"][number];

function makeDreamPart(overrides?: { data?: unknown }): Part {
  return {
    type: DREAM_OPERATIONS_PART_TYPE,
    data: overrides?.data ?? {
      snapshot: {
        writes: [],
        proposals: [],
        demotions: [],
        entity_invalidations: [],
      },
      dream_pass_id: "dp-1",
      user_id: "u-1",
    },
  } as unknown as Part;
}

describe("dream operations stream part", () => {
  it("recognises the wire type", () => {
    expect(isDreamOperationsPart(makeDreamPart())).toBe(true);
  });

  it("ignores unrelated part types", () => {
    const textPart = { type: "text", text: "hi" } as unknown as Part;
    expect(isDreamOperationsPart(textPart)).toBe(false);
  });

  it("returns the typed payload when shape is valid", () => {
    const part = makeDreamPart();
    const data = readDreamOperationsPart(part);
    expect(data).not.toBeNull();
    expect(data?.dream_pass_id).toBe("dp-1");
    expect(data?.user_id).toBe("u-1");
    expect(data?.snapshot.writes).toEqual([]);
  });

  it("returns null for malformed payloads instead of throwing", () => {
    // Missing dream_pass_id — handler must swallow without crashing so a
    // single bad event can't break the surrounding chat stream parse.
    const bad = makeDreamPart({
      data: { snapshot: {}, user_id: "u-1" },
    });
    expect(readDreamOperationsPart(bad)).toBeNull();
  });

  it("handler logs and returns without throwing for valid events", () => {
    const debugSpy = vi.spyOn(console, "debug").mockImplementation(() => {});
    expect(() => handleDreamOperationsPart(makeDreamPart())).not.toThrow();
    expect(debugSpy).toHaveBeenCalledWith(
      "[copilot] dream.operations event received",
      "dp-1",
    );
    debugSpy.mockRestore();
  });

  it("handler silently no-ops on a non-dream part", () => {
    const textPart = { type: "text", text: "hi" } as unknown as Part;
    const debugSpy = vi.spyOn(console, "debug").mockImplementation(() => {});
    expect(() => handleDreamOperationsPart(textPart)).not.toThrow();
    expect(debugSpy).not.toHaveBeenCalled();
    debugSpy.mockRestore();
  });
});
