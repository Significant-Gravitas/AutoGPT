import type { ExpertSetupItem } from "@/app/api/__generated__/models/expertSetupItem";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { Calendar03Icon } from "@hugeicons/core-free-icons";

interface Props {
  item: ExpertSetupItem;
  isConnectable: boolean;
  isGranting: boolean;
  onConnect: () => void;
  onAllow: () => void;
}

export function SetupNeededRow({
  item,
  isConnectable,
  isGranting,
  onConnect,
  onAllow,
}: Props) {
  const provider = item.providers[0];
  const providerName = provider ? formatProviderName(provider) : "A service";
  const workflowName = item.workflow_name ?? "a workflow";
  const isCredential = item.resolution !== "workflow";

  return (
    <article className="flex items-center gap-2.5 rounded-2xl bg-white px-3.5 py-2.5 smooth-shadow-ring-sm">
      {isCredential && provider ? (
        <IntegrationLogo provider={provider} size={20} className="shrink-0" />
      ) : (
        <Icon
          icon={Calendar03Icon}
          size={18}
          className="shrink-0 text-zinc-500"
          aria-hidden="true"
        />
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <Text
          variant="body-medium"
          as="span"
          className="truncate !text-zinc-800"
        >
          {isCredential
            ? `${providerName} for ${item.expert_name}`
            : `Schedule ${workflowName} for ${item.expert_name}`}
        </Text>
        <Text variant="small" as="span" className="truncate !text-zinc-400">
          {isCredential
            ? `${workflowName} needs it to run on schedule.`
            : "Every connection is in place; the schedule still has to be created."}
        </Text>
      </div>
      <div className="flex shrink-0 items-center">
        {item.resolution === "allow" ? (
          <Button
            variant="secondary"
            size="small"
            loading={isGranting}
            onClick={onAllow}
          >
            Allow
          </Button>
        ) : item.resolution === "connect" ? (
          isConnectable ? (
            <Button variant="primary" size="small" onClick={onConnect}>
              Connect
            </Button>
          ) : (
            <Text variant="small" as="span" className="!text-zinc-500">
              Needs a platform key
            </Text>
          )
        ) : item.library_agent_id ? (
          <Button
            as="NextLink"
            href={`/library/agents/${item.library_agent_id}`}
            variant="outline"
            size="small"
          >
            Open workflow
          </Button>
        ) : null}
      </div>
    </article>
  );
}
