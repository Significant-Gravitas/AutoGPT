import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import { ProviderMetadataSupportedAuthTypesItem as AuthType } from "@/app/api/__generated__/models/providerMetadataSupportedAuthTypesItem";

import { formatProviderName } from "../../helpers";

export type AuthMethod = (typeof AuthType)[keyof typeof AuthType];

export { AuthType };

export interface ConnectableProvider {
  id: string;
  name: string;
  description?: string | null;
  supportedAuthTypes: AuthMethod[];
}

const KNOWN_AUTH_METHODS: ReadonlySet<AuthMethod> = new Set(
  Object.values(AuthType),
);

function normalizeAuthTypes(
  raw: readonly AuthMethod[] | undefined,
): AuthMethod[] {
  if (!raw) return [];
  return raw.filter((t) => KNOWN_AUTH_METHODS.has(t));
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
      supportedAuthTypes: normalizeAuthTypes(item.supported_auth_types),
    });
  }
  result.sort((a, b) => a.name.localeCompare(b.name));
  return result;
}

function normalize(text: string): string {
  return text.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();
}

export function filterConnectableProviders(
  providers: ConnectableProvider[],
  query: string,
): ConnectableProvider[] {
  const q = normalize(query.trim());
  if (!q) return providers;
  return providers.filter((p) => {
    if (normalize(p.name).includes(q)) return true;
    if (normalize(p.id).includes(q)) return true;
    if (p.description && normalize(p.description).includes(q)) return true;
    return false;
  });
}
