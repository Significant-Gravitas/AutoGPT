import type { ConnectableProvider } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/helpers";

// What the section shows when the model picked nothing — a thin
// transcript, a skipped dump, or a job that never landed. An empty
// "Connect your tools" panel reads as a broken dialog, and these are what
// most people wire up first anyway.
const FALLBACK_PROVIDER_IDS = [
  "google",
  "slack",
  "notion",
  "github",
  "discord",
  "todoist",
];
const FALLBACK_COUNT = 6;

// The preferred ids in order, padded from the rest of the registry so a
// deployment missing some of them still fills the section.
export function fallbackProviders(all: ConnectableProvider[]) {
  const byPreference = FALLBACK_PROVIDER_IDS.map((id) =>
    all.find((provider) => provider.id === id),
  ).filter((provider) => provider !== undefined);
  const picked = new Set(byPreference.map((provider) => provider.id));
  const padding = all.filter((provider) => !picked.has(provider.id));
  return [...byPreference, ...padding].slice(0, FALLBACK_COUNT);
}
