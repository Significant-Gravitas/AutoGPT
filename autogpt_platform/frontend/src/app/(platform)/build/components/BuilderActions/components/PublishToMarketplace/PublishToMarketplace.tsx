import { ShareIcon } from "@phosphor-icons/react";
import { BuilderActionButton } from "../BuilderActionButton";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { usePublishToMarketplace } from "./usePublishToMarketplace";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";

export const PublishToMarketplace = ({ flowID }: { flowID: string | null }) => {
  const { handlePublishToMarketplace, publishState, handleStateChange } =
    usePublishToMarketplace({ flowID });

  return (
    <>
      <Tooltip>
        <TooltipTrigger asChild>
          <BuilderActionButton
            onClick={handlePublishToMarketplace}
            disabled={!flowID}
          >
            <ShareIcon className="size-6 drop-shadow-sm" />
          </BuilderActionButton>
        </TooltipTrigger>
        <TooltipContent>Publish to Marketplace</TooltipContent>
      </Tooltip>

      <PublishAgentModal
        targetState={publishState}
        onStateChange={handleStateChange}
        preSelectedAgentId={flowID || undefined}
      />
    </>
  );
};
