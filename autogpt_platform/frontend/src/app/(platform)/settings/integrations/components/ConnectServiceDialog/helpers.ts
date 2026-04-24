import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";

import { formatProviderName } from "../../helpers";

export interface ConnectableProvider {
  id: string;
  name: string;
  description?: string | null;
}

export function toConnectableProviders(
  metadata: ProviderMetadata[],
): ConnectableProvider[] {
  const seen = new Set<string>();
  const result: ConnectableProvider[] = [];
  for (const item of metadata) {
    if (seen.has(item.name)) continue;
    seen.add(item.name);
    result.push({
      id: item.name,
      name: formatProviderName(item.name),
      description: item.description,
    });
  }
  result.sort((a, b) => a.name.localeCompare(b.name));
  return result;
}

export function filterConnectableProviders(
  providers: ConnectableProvider[],
  query: string,
): ConnectableProvider[] {
  const q = query.trim().toLowerCase();
  if (!q) return providers;
  return providers.filter((p) => {
    if (p.name.toLowerCase().includes(q)) return true;
    if (p.id.toLowerCase().includes(q)) return true;
    if (p.description?.toLowerCase().includes(q)) return true;
    return false;
  });
}
