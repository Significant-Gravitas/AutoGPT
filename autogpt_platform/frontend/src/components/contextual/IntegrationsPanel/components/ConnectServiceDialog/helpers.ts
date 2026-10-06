import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import { ProviderMetadataSupportedAuthTypesItem as AuthType } from "@/app/api/__generated__/models/providerMetadataSupportedAuthTypesItem";

import { serviceIcon, serviceKey, serviceName } from "../../helpers";

export type AuthMethod = (typeof AuthType)[keyof typeof AuthType];

export { AuthType };

export interface ConnectableProvider {
  id: string;
  serviceId: string;
  name: string;
  description?: string | null;
  supportedAuthTypes: AuthMethod[];
  authProviderByType?: Partial<Record<AuthMethod, string>>;
  searchTerms?: string[];
  mcpServer?: ProviderMetadata["mcp_server"];
  iconId?: string;
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
  const byService = new Map<string, ConnectableProvider>();
  for (const item of metadata) {
    if (seen.has(item.name)) continue;
    seen.add(item.name);

    const displayProvider = item.name === "codex" ? "openai" : item.name;
    const ref = {
      provider: item.name,
      service: item.service,
      service_name: item.service_name,
      service_icon: item.service_icon,
    };
    const key = serviceKey(ref);
    const authTypes = normalizeAuthTypes(item.supported_auth_types);
    const existing = byService.get(key);
    const provider: ConnectableProvider = existing ?? {
      id: item.mcp_server ? item.name : displayProvider,
      serviceId: key,
      name: serviceName(ref),
      description: item.description,
      supportedAuthTypes: [],
      iconId: serviceIcon(ref),
    };
    if (item.mcp_server) {
      provider.mcpServer = item.mcp_server;
      if (existing) {
        provider.searchTerms = Array.from(
          new Set([...(provider.searchTerms ?? []), item.name]),
        );
      }
    } else if (existing?.mcpServer && provider.id !== displayProvider) {
      provider.searchTerms = Array.from(
        new Set([...(provider.searchTerms ?? []), provider.id]),
      );
      provider.id = displayProvider;
    }

    for (const authType of authTypes) {
      const alreadySupported = provider.supportedAuthTypes.includes(authType);
      if (!alreadySupported) provider.supportedAuthTypes.push(authType);
      if (item.name === displayProvider) {
        delete provider.authProviderByType?.[authType];
      } else if (!alreadySupported) {
        provider.authProviderByType = {
          ...provider.authProviderByType,
          [authType]: item.name,
        };
      }
    }
    if (item.name !== displayProvider && !item.mcp_server) {
      provider.searchTerms = Array.from(
        new Set([...(provider.searchTerms ?? []), item.name]),
      );
    }
    if (item.name === displayProvider && !item.mcp_server) {
      provider.description = item.description;
    }
    byService.set(key, provider);
  }

  const openai = byService.get("openai");
  if (openai?.authProviderByType?.oauth2 === "codex") {
    openai.description =
      "OpenAI models via API key or your ChatGPT subscription";
  }

  const result = Array.from(byService.values());
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
