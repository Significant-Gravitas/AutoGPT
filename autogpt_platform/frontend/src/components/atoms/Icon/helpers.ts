import { getAutoGPTIcon } from "./agptIcons";
import { iconRegistry } from "./registry";

// All-or-nothing switch. We only render the AutoGPT icon set when the optional
// `@autogpt/icons` package is installed AND every icon in the registry resolves
// from it. If even one is missing (package absent, or a version that dropped an
// icon), the whole app falls back to Phosphor — so the UI never mixes the two
// sets.
export function areAutoGPTIconsAvailable() {
  return Object.values(iconRegistry).every(
    (entry) => getAutoGPTIcon(entry.autogpt) !== undefined,
  );
}
