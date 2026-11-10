import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { cn } from "@/lib/utils";
import { WorkflowSquare01Icon } from "@hugeicons/core-free-icons";
import {
  CARD,
  InfoEmpty,
  InfoGroup,
  InfoList,
  InfoLoading,
  InfoPill,
  InfoRow,
} from "./InfoParts";

interface CredentialsProps {
  credentials: ExpertCredentialRef[];
  isLoading: boolean;
}

export function ConnectionsGroup({ credentials, isLoading }: CredentialsProps) {
  return (
    <InfoGroup label="Connections" count={credentials.length}>
      {isLoading ? (
        <InfoLoading />
      ) : credentials.length === 0 ? (
        <InfoEmpty>No connections granted yet.</InfoEmpty>
      ) : (
        <InfoList>
          {credentials.map((credential) => {
            const service = credential.service_name || credential.provider;
            return (
              <InfoRow
                key={credential.credential_id}
                tile={
                  <IntegrationLogo
                    provider={credential.provider}
                    alt={service}
                    size={16}
                  />
                }
                title={service}
                subtitle={
                  credential.title !== service ? credential.title : undefined
                }
              />
            );
          })}
        </InfoList>
      )}
    </InfoGroup>
  );
}

export function SkillsGroup({ skills }: { skills: string[] }) {
  return (
    <InfoGroup label="Skills" count={skills.length}>
      {skills.length === 0 ? (
        <InfoEmpty>No skills listed yet.</InfoEmpty>
      ) : (
        <ul className={cn(CARD, "flex flex-wrap gap-1.5 p-2.5")}>
          {skills.map((skill) => (
            <li
              key={skill}
              className="inline-flex h-7 items-center rounded-lg bg-surface-tray px-2.5"
            >
              <Text variant="inbox" as="span" tone="text">
                {skill}
              </Text>
            </li>
          ))}
        </ul>
      )}
    </InfoGroup>
  );
}

export function WorkflowsGroup({
  workflows,
}: {
  workflows: Expert["workflows"];
}) {
  const named = workflows.filter((workflow) => workflow.name);
  return (
    <InfoGroup label="Workflows" count={named.length}>
      {named.length === 0 ? (
        <InfoEmpty>No workflows yet.</InfoEmpty>
      ) : (
        <InfoList>
          {named.map((workflow) => (
            <InfoRow
              key={workflow.id}
              tile={<Icon icon={WorkflowSquare01Icon} size={14} />}
              title={workflow.name ?? ""}
              subtitle={workflow.description ?? undefined}
              trailing={
                workflow.schedule_id || workflow.schedule_cron ? (
                  <InfoPill>Scheduled</InfoPill>
                ) : null
              }
            />
          ))}
        </InfoList>
      )}
    </InfoGroup>
  );
}
