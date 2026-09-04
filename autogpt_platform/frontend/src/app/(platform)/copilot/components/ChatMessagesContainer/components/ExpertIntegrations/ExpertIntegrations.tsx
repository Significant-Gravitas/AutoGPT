"use client";

import Link from "next/link";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import {
  formatCredentialName,
  formatProviderName,
} from "@/components/contextual/IntegrationsPanel/helpers";
import { useExpertIntegrations } from "./useExpertIntegrations";

interface Props {
  expertId: string;
  expertName: string;
}

const VISIBLE_LOGOS = 3;

/** The integrations this expert can reach, under its name in the thread header.
 *
 * Renders nothing when the expert has none: a header that says "Integrations"
 * above an empty row reads as a loading failure rather than an empty state.
 */
export function ExpertIntegrations({ expertId, expertName }: Props) {
  const { integrations } = useExpertIntegrations(expertId);

  if (integrations.length === 0) return null;

  const visible = integrations.slice(0, VISIBLE_LOGOS);
  const overflow = integrations.length - visible.length;

  return (
    <Popover>
      <PopoverTrigger
        className="flex shrink-0 items-center gap-1 rounded-full py-1 pl-1.5 pr-3 transition-colors hover:bg-zinc-100/80"
        aria-label={`${expertName}'s integrations`}
        data-testid="expert-integrations"
      >
        {visible.map((integration) => (
          <IntegrationLogo
            key={integration.credential_id}
            provider={integration.provider}
            alt={formatProviderName(integration.provider)}
          />
        ))}
        {overflow > 0 ? (
          <span className="ml-0.5 text-xs text-zinc-500">+{overflow}</span>
        ) : null}
      </PopoverTrigger>
      <PopoverContent align="start" className="w-64 p-3">
        <div className="mb-2 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
          Integrations
        </div>
        <ul className="flex flex-col gap-2">
          {integrations.map((integration) => (
            <li
              key={integration.credential_id}
              className="flex items-center gap-2"
            >
              <IntegrationLogo provider={integration.provider} />
              <span className="truncate text-sm text-zinc-700">
                {formatCredentialName(integration.title, integration.provider)}
              </span>
            </li>
          ))}
        </ul>
        <Link
          href={`/team/${expertId}`}
          className="mt-3 block text-xs text-zinc-500 underline hover:text-zinc-800"
        >
          Manage what {expertName} can access
        </Link>
      </PopoverContent>
    </Popover>
  );
}
