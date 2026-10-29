import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { ExpertSection } from "./ExpertSection";

interface Props {
  name: string;
  providers: string[];
}

export function ExpertAccess({ name, providers }: Props) {
  if (providers.length === 0) return null;

  return (
    <ExpertSection
      title={`Services ${name} can work with`}
      description={`${name} will ask you to connect these the first time they are needed — nothing is shared until you do.`}
    >
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {providers.map((provider) => (
          <li
            key={provider}
            className="flex min-w-0 items-center gap-3 rounded-2xl bg-surface p-3 smooth-shadow-ring-sm"
          >
            <span
              aria-hidden="true"
              className="flex size-10 shrink-0 items-center justify-center rounded-xl border border-border-base bg-gradient-to-b from-surface to-surface-muted"
            >
              <IntegrationLogo provider={provider} size={22} />
            </span>
            <span className="min-w-0 break-words text-sm font-medium leading-5 text-foreground-emphasis">
              {formatProviderName(provider)}
            </span>
          </li>
        ))}
      </ul>
    </ExpertSection>
  );
}
