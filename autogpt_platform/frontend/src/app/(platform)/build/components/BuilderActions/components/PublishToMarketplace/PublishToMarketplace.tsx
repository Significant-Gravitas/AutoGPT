import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import { ShareIcon } from "@phosphor-icons/react";
import { usePublishToMarketplace } from "./usePublishToMarketplace";

export const PublishToMarketplace = ({ flowID }: { flowID: string | null }) => {
  const { handlePublishToMarketplace, publishState, handleStateChange } =
    usePublishToMarketplace({ flowID });

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            size="icon"
            onClick={handlePublishToMarketplace}
            disabled={!flowID}
          >
            <ShareIcon className="size-4" />
          </Button>
        </TooltipTrigger>
        <TooltipContent>Publish to Marketplace</TooltipContent>
      </Tooltip>

      <PublishAgentModal
        targetState={publishState}
        onStateChange={handleStateChange}
        preSelectedAgentId={flowID || undefined}
        showTrigger={false}
      />
    </>
  );
};
