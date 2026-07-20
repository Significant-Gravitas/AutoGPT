import { getAutoGPTIcon } from "./agptIcons";
import { iconMap } from "./iconMap";
import { iconRegistry } from "./registry";

let cached: boolean | undefined;

// Every AutoGPT export the app can render: the Phosphor compat mappings
// (`iconMap`, used by createIcon) plus the semantic `Icon` atom's registry.
// Both must resolve, or the availability check would pass while an unmapped
// registry icon silently falls back to Phosphor — mixing the two sets.
const requiredAutoGPTIcons = [
  ...Object.values(iconMap),
  ...Object.values(iconRegistry).map((entry) => entry.autogpt),
];

// All-or-nothing switch. We only render the AutoGPT icon set when the optional
// `@autogpt/icons` package is installed AND every required icon resolves from
// it. If even one is missing (package absent, or a version that dropped an
// icon), the whole app falls back to Phosphor — so the UI never mixes the two
// sets. Availability is fixed for the lifetime of the bundle, so it's cached.
export function areAutoGPTIconsAvailable() {
  if (cached === undefined) {
    cached = requiredAutoGPTIcons.every(
      (name) => getAutoGPTIcon(name) !== undefined,
    );
  }
  return cached;
}

// Test-only: clears the memoized availability result so mocks can be re-read.
export function resetAutoGPTIconsAvailabilityCache() {
  cached = undefined;
}
