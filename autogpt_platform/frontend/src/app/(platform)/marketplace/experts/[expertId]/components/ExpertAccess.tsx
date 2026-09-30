import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { ExpertPill } from "./ExpertPill";
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
          <ExpertPill
            key={provider}
            icon={<IntegrationLogo provider={provider} />}
            label={formatProviderName(provider)}
          />
        ))}
      </ul>
    </ExpertSection>
  );
}
