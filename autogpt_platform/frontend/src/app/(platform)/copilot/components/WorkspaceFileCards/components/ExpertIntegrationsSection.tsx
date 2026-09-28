"use client";

import { PlugSocketIcon } from "@hugeicons/core-free-icons";
import type { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import {
  formatCredentialName,
  formatProviderName,
} from "@/components/contextual/IntegrationsPanel/helpers";
import { StackSection } from "./StackSection";

interface Props {
  integrations: ExpertCredentialRef[];
}

export function ExpertIntegrationsSection({ integrations }: Props) {
  if (integrations.length === 0) return null;

  return (
    <StackSection
      title="Integrations"
      icon={PlugSocketIcon}
      count={integrations.length}
    >
      <ul
        data-testid="expert-integrations-card"
        className="-mx-2.5 grid max-h-72 gap-0.5 overflow-y-auto px-2.5 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200"
      >
        {integrations.map((integration) => (
          <li
            key={integration.credential_id}
            className="-mx-2.5 flex items-center gap-3 rounded-xl px-2.5 py-1.5"
          >
            <IntegrationLogo
              provider={integration.provider}
              alt={formatProviderName(integration.provider)}
              className="shrink-0"
            />
            <span className="truncate text-sm text-zinc-800">
              {formatCredentialName(integration.title, integration.provider)}
            </span>
          </li>
        ))}
      </ul>
    </StackSection>
  );
}
