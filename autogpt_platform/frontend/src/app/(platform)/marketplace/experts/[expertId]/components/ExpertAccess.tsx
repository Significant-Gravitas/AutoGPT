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
      title={`Access ${name} will ask for`}
      description={`Connect these when ${name} first needs them — nothing is shared until you do.`}
    >
      <ul className="flex flex-wrap gap-2">
        {providers.map((provider) => (
          <li
            key={provider}
            className="flex items-center gap-2 rounded-lg bg-white px-2.5 py-1.5 text-sm text-zinc-700 ring-1 ring-inset ring-zinc-200/80"
          >
            <IntegrationLogo
              provider={provider}
              alt={formatProviderName(provider)}
            />
            {formatProviderName(provider)}
          </li>
        ))}
      </ul>
    </ExpertSection>
  );
}
