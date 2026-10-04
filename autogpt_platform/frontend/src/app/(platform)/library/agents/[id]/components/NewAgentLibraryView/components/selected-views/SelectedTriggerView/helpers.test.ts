import { describe, expect, test } from "vitest";
import { mergePresetInputs, splitPresetInputs } from "./helpers";

// A triggered preset stores regular graph inputs at the top level and the
// trigger block's config nested under `_node_input_mask_{node prefix}`.
const STORED = {
  topic: "weather",
  _node_input_mask_abc123: { repo: "owner/repo", events: ["push"] },
};

describe("splitPresetInputs", () => {
  test("separates the trigger config from the regular graph inputs", () => {
    const split = splitPresetInputs(STORED);
    expect(split.inputs).toEqual({ topic: "weather" });
    expect(split.triggerConfig).toEqual({
      repo: "owner/repo",
      events: ["push"],
    });
    expect(split.maskKey).toBe("_node_input_mask_abc123");
  });

  test("treats a preset with no mask key as regular inputs only", () => {
    const split = splitPresetInputs({ topic: "weather" });
    expect(split.inputs).toEqual({ topic: "weather" });
    expect(split.triggerConfig).toEqual({});
    expect(split.maskKey).toBeNull();
  });

  test("tolerates null/undefined inputs", () => {
    expect(splitPresetInputs(null).inputs).toEqual({});
    expect(splitPresetInputs(undefined).maskKey).toBeNull();
  });

  test("ignores a non-object value under the mask key", () => {
    const split = splitPresetInputs({ _node_input_mask_abc123: "not-a-dict" });
    expect(split.triggerConfig).toEqual({});
    expect(split.maskKey).toBe("_node_input_mask_abc123");
  });
});

describe("mergePresetInputs", () => {
  test("round-trips an untouched preset back to the stored shape", () => {
    expect(mergePresetInputs(splitPresetInputs(STORED))).toEqual(STORED);
  });

  test("re-nests an edited trigger config under the original mask key", () => {
    const split = splitPresetInputs(STORED);
    const edited = {
      ...split,
      triggerConfig: { ...split.triggerConfig, repo: "owner/other" },
    };
    expect(mergePresetInputs(edited)).toEqual({
      topic: "weather",
      _node_input_mask_abc123: { repo: "owner/other", events: ["push"] },
    });
  });

  test("keeps edited graph inputs alongside the trigger config", () => {
    const split = splitPresetInputs(STORED);
    const edited = { ...split, inputs: { topic: "sports" } };
    expect(mergePresetInputs(edited)).toEqual({
      topic: "sports",
      _node_input_mask_abc123: { repo: "owner/repo", events: ["push"] },
    });
  });

  test("does not invent a mask key when the preset never had one", () => {
    const merged = mergePresetInputs({
      inputs: { topic: "weather" },
      triggerConfig: {},
      maskKey: null,
    });
    expect(merged).toEqual({ topic: "weather" });
  });
});
