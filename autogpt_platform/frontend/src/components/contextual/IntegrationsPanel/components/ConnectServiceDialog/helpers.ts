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
  authProviderByType?: Partial<Record<AuthMethod, string>>;
  searchTerms?: string[];
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
  const byDisplayProvider = new Map<string, ConnectableProvider>();
  for (const item of metadata) {
    if (seen.has(item.name)) continue;
    seen.add(item.name);

    // Which entry this credential is filed under. A ChatGPT subscription is
    // a different credential from an OpenAI API key and the same entry to a
    // person looking for it -- the server says which, because the client
    // cannot be told when a provider is added.
    const displayProvider = item.display_alias ?? item.name;
    const authTypes = normalizeAuthTypes(item.supported_auth_types);
    const existing = byDisplayProvider.get(displayProvider);
    const provider = existing ?? {
      id: displayProvider,
      name: formatProviderName(displayProvider),
      description: item.description,
      supportedAuthTypes: [],
    };

    for (const authType of authTypes) {
      const alreadySupported = provider.supportedAuthTypes.includes(authType);
      if (!alreadySupported) {
        provider.supportedAuthTypes.push(authType);
      }
      if (item.name === displayProvider) {
        delete provider.authProviderByType?.[authType];
      } else if (!alreadySupported) {
        provider.authProviderByType = {
          ...provider.authProviderByType,
          [authType]: item.name,
        };
      }
    }
    if (item.name !== displayProvider) {
      provider.searchTerms = Array.from(
        new Set([...(provider.searchTerms ?? []), item.name]),
      );
    }
    if (item.name === displayProvider) {
      provider.description = item.description;
    }
    byDisplayProvider.set(displayProvider, provider);
  }

  const result = Array.from(byDisplayProvider.values());
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
    if (p.searchTerms?.some((term) => normalize(term).includes(q))) return true;
    if (p.description && normalize(p.description).includes(q)) return true;
    return false;
  });
}
