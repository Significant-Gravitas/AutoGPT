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
      <ul className="flex flex-wrap gap-2">
        {providers.map((provider) => (
          <li
            key={provider}
            className="flex min-w-0 items-center gap-2 rounded-lg bg-white px-2.5 py-1.5 text-sm text-zinc-700 ring-1 ring-inset ring-zinc-200/80"
          >
            {/* The label beside it already names the provider; without this a
                screen reader announces the name twice. */}
            <span aria-hidden="true" className="flex shrink-0">
              <IntegrationLogo provider={provider} />
            </span>
            <span className="truncate">{formatProviderName(provider)}</span>
          </li>
        ))}
      </ul>
    </ExpertSection>
  );
}
