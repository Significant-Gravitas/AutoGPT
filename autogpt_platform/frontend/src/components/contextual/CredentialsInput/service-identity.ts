import type { BlockIOCredentialsSubSchema } from "@/lib/autogpt-server-api/types";
import { getHostFromUrl } from "@/lib/utils/url";

const SERVICES: Partial<
  Record<
    string,
    {
      name: string;
      icon: string;
      keyHelp: string;
      bearer: boolean;
    }
  >
> = {
  "api.linear.app": {
    name: "Linear",
    icon: "linear",
    keyHelp:
      "In Linear, open Settings → Account → Security & Access and create a personal API key. Give it access to the teams this workflow needs.",
    bearer: false,
  },
  "api.github.com": {
    name: "GitHub",
    icon: "github",
    keyHelp:
      "Create a GitHub personal access token with access to the repositories and permissions this workflow needs.",
    bearer: true,
  },
};

export function credentialHosts(schema: BlockIOCredentialsSubSchema) {
  if (!schema.credentials_types?.includes("host_scoped")) return [];
  return [
    ...new Set(
      (schema.discriminator_values ?? []).flatMap((url) => {
        const host = getHostFromUrl(url);
        return host ? [host.toLowerCase()] : [];
      }),
    ),
  ].sort();
}

export function credentialService(schema: BlockIOCredentialsSubSchema) {
  if (schema.credentials_provider[0] !== "http") return undefined;
  const hosts = credentialHosts(schema);
  if (hosts.length !== 1) return undefined;
  return { host: hosts[0], ...SERVICES[hosts[0]] };
}
