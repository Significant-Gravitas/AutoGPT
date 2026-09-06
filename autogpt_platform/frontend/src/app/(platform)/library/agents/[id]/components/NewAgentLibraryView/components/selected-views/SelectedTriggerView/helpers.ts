// Mirrors the backend's `node_input_mask_key()`: a triggered preset nests the
// trigger block's config under a per-node key, alongside its regular graph
// inputs. The node id isn't exposed to the client, so the key is found by prefix
// (a graph has at most one trigger node) and preserved on save rather than
// reconstructed.
const NODE_INPUT_MASK_PREFIX = "_node_input_mask_";

export type SplitPresetInputs = {
  inputs: Record<string, any>;
  triggerConfig: Record<string, any>;
  maskKey: string | null;
};

export function splitPresetInputs(
  presetInputs: Record<string, any> | null | undefined,
): SplitPresetInputs {
  const all = presetInputs ?? {};
  const maskKey =
    Object.keys(all).find((key) => key.startsWith(NODE_INPUT_MASK_PREFIX)) ??
    null;

  if (!maskKey) return { inputs: { ...all }, triggerConfig: {}, maskKey: null };

  const { [maskKey]: mask, ...inputs } = all;
  return {
    inputs,
    triggerConfig:
      mask && typeof mask === "object" && !Array.isArray(mask) ? mask : {},
    maskKey,
  };
}

export function mergePresetInputs({
  inputs,
  triggerConfig,
  maskKey,
}: SplitPresetInputs): Record<string, any> {
  if (!maskKey) return { ...inputs };
  return { ...inputs, [maskKey]: triggerConfig };
}
