import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import { usePublishToMarketplace } from "./usePublishToMarketplace";
import { Share01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useBuilderTenantScope } from "@/app/(platform)/build/hooks/useBuilderTenantScope";

interface Props {
  flowID: string | null;
  flowVersion: number | null;
}

export function PublishToMarketplace({ flowID, flowVersion }: Props) {
  const tenantScope = useBuilderTenantScope();
  const { handlePublishToMarketplace, publishState, handleStateChange } =
    usePublishToMarketplace({ flowID, flowVersion });

  const isDisabled = !flowID || flowVersion === null;

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            size="icon"
            onClick={handlePublishToMarketplace}
            disabled={isDisabled}
          >
            <Icon icon={Share01Icon} className="size-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Publish to Marketplace</TooltipContent>
      </Tooltip>

      <PublishAgentModal
        targetState={publishState}
        onStateChange={handleStateChange}
        preSelectedAgentId={flowID || undefined}
        preSelectedAgentVersion={flowVersion ?? undefined}
        preSelectedOrganizationId={tenantScope.organizationId}
        preSelectedTeamId={tenantScope.teamId}
        showTrigger={false}
      />
    </>
  );
}
