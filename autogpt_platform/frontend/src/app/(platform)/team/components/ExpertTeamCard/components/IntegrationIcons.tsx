"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import Image from "next/image";
import { useState } from "react";

const VISIBLE_LOGOS = 3;

interface Props {
  providers: string[];
}

/** The services an expert can reach, as the logos the connections dialog
 *  uses. Past three, a "+N more" names the rest on hover. */
export function IntegrationIcons({ providers }: Props) {
  if (providers.length === 0) return null;
  const shown = providers.slice(0, VISIBLE_LOGOS);
  const hidden = providers.slice(VISIBLE_LOGOS);

  return (
    <ul aria-label="Integrations" className="flex items-center gap-1.5">
      {shown.map((provider) => (
        <li key={provider} className="flex">
          <ProviderLogo provider={provider} />
        </li>
      ))}
      {hidden.length > 0 ? (
        <li className="flex">
          <Tooltip>
            <TooltipTrigger asChild>
              <span tabIndex={0} className="cursor-default rounded-sm">
                <Text
                  variant="body-medium"
                  as="span"
                  className="whitespace-nowrap !text-zinc-800"
                >
                  +{hidden.length} more
                </Text>
              </span>
            </TooltipTrigger>
            <TooltipContent>
              {hidden.map(formatProviderName).join(", ")}
            </TooltipContent>
          </Tooltip>
        </li>
      ) : null}
    </ul>
  );
}

interface LogoProps {
  provider: string;
}

function ProviderLogo({ provider }: LogoProps) {
  const [isBroken, setIsBroken] = useState(false);
  const name = formatProviderName(provider);

  if (isBroken) {
    return (
      <span
        role="img"
        aria-label={name}
        className="flex size-4 shrink-0 items-center justify-center rounded bg-zinc-100 text-[9px] font-semibold uppercase leading-none text-zinc-600"
      >
        {name.charAt(0)}
      </span>
    );
  }
  return (
    <Image
      src={`/integrations/${provider}.png`}
      alt={name}
      width={16}
      height={16}
      loading="lazy"
      className="size-4 shrink-0 object-contain"
      onError={() => setIsBroken(true)}
    />
  );
}
