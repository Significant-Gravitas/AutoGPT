"use client";

import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import {
  formatCredentialName,
  formatProviderName,
} from "@/components/contextual/IntegrationsPanel/helpers";
import { useExpertIntegrations } from "./useExpertIntegrations";

interface Props {
  expertId: string;
}

/** The integrations this expert can reach, listed down the gutter under the
 *  thread chip. Only wide screens have a gutter; narrower ones would lay the
 *  list over the messages, and the files card lists them there anyway. */
export function ExpertIntegrations({ expertId }: Props) {
  const { integrations } = useExpertIntegrations(expertId);

  if (integrations.length === 0) return null;

  return (
    <ul
      aria-label="Integrations"
      data-testid="expert-integrations"
      className="hidden flex-col items-start gap-1.5 pl-1.5 xl:flex"
    >
      {integrations.map((integration) => (
        <li
          key={integration.credential_id}
          className="pointer-events-auto flex max-w-[12rem] items-center gap-2 rounded-full border border-zinc-200/70 bg-white/75 py-1 pl-1.5 pr-3 shadow-sm backdrop-blur-md"
        >
          <IntegrationLogo
            provider={integration.provider}
            alt={formatProviderName(integration.provider)}
          />
          <span className="truncate text-xs text-zinc-600">
            {formatCredentialName(integration.title, integration.provider)}
          </span>
        </li>
      ))}
    </ul>
  );
}
