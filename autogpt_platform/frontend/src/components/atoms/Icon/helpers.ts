import { getAutoGPTIcon } from "./agptIcons";
import { iconMap } from "./iconMap";

let cached: boolean | undefined;

// All-or-nothing switch. We only render the AutoGPT icon set when the optional
// `@autogpt/icons` package is installed AND every mapped icon resolves from it.
// If even one is missing (package absent, or a version that dropped an icon),
// the whole app falls back to Phosphor — so the UI never mixes the two sets.
// Availability is fixed for the lifetime of the bundle, so the result is cached.
export function areAutoGPTIconsAvailable() {
  if (cached === undefined) {
    cached = Object.values(iconMap).every(
      (name) => getAutoGPTIcon(name) !== undefined,
    );
  }
  return cached;
}

// Test-only: clears the memoized availability result so mocks can be re-read.
export function resetAutoGPTIconsAvailabilityCache() {
  cached = undefined;
}
