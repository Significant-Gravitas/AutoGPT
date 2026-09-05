"use client";

import { ExpertWorkflowChainItem } from "@/app/api/__generated__/models/expertWorkflowChainItem";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { cn } from "@/lib/utils";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { toDisplayName } from "@/components/renderers/InputRenderer/custom/CredentialField/helpers";

const MAX_VISIBLE = 3;

export function getWorkflowCredentialProviders(
  agent: LibraryAgent | undefined,
  chain: ExpertWorkflowChainItem[],
): string[] {
  const properties = agent?.credentials_input_schema?.properties;
  const fromSchema = properties
    ? Object.values(properties as Record<string, unknown>).flatMap((schema) => {
        const providers =
          schema &&
          typeof schema === "object" &&
          "credentials_provider" in schema
            ? (schema as { credentials_provider?: unknown })
                .credentials_provider
            : undefined;
        return Array.isArray(providers) && providers.length === 1
          ? providers.filter((p): p is string => typeof p === "string")
          : [];
      })
    : [];
  const fromChain = chain
    .map((item) => item.provider)
    .filter((provider): provider is string => Boolean(provider));
  return [...new Set([...fromSchema, ...fromChain])];
}

interface Props {
  providers: string[];
}

export function WorkflowCredentialStack({ providers }: Props) {
  if (providers.length === 0) return null;
  const visible = providers.slice(0, MAX_VISIBLE);
  const hidden = providers.length - visible.length;
  return (
    <div
      className="flex items-center"
      role="list"
      aria-label="Credentials used"
      title={providers.map(toDisplayName).join(", ")}
    >
      {visible.map((provider, index) => (
        <span
          key={provider}
          role="listitem"
          className={cn(
            "flex size-7 items-center justify-center rounded-full border border-zinc-200 bg-white ring-2 ring-white",
            index > 0 && "-ml-2",
            ["z-30", "z-20", "z-10"][index],
          )}
        >
          <IntegrationLogo
            provider={provider}
            alt={toDisplayName(provider)}
            size={16}
          />
        </span>
      ))}
      {hidden > 0 ? (
        <span
          role="listitem"
          className="-ml-2 flex size-7 items-center justify-center rounded-full border border-zinc-200 bg-zinc-100 text-[11px] font-medium text-zinc-600 ring-2 ring-white"
          aria-label={`${hidden} more`}
        >
          +{hidden}
        </span>
      ) : null}
    </div>
  );
}
