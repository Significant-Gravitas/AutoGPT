import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn, beautifyString } from "@/lib/utils";
import { CustomNodeData } from "../../CustomNode";
import { useSubAgentUpdateState } from "./useSubAgentUpdateState";
import { IncompatibleUpdateDialog } from "./components/IncompatibleUpdateDialog";
import { ResolutionModeBar } from "./components/ResolutionModeBar";
import { Alert01Icon, ArrowUp02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

/**
 * Inline component for the update bar that can be placed after the header.
 * Use this inside the node content where you want the bar to appear.
 */
type SubAgentUpdateFeatureProps = {
  nodeID: string;
  nodeData: CustomNodeData;
};

export function SubAgentUpdateFeature({
  nodeID,
  nodeData,
}: SubAgentUpdateFeatureProps) {
  const {
    updateInfo,
    isInResolutionMode,
    handleUpdateClick,
    showIncompatibilityDialog,
    setShowIncompatibilityDialog,
    handleConfirmIncompatibleUpdate,
  } = useSubAgentUpdateState({ nodeID: nodeID, nodeData: nodeData });

  const agentName = nodeData.title || "Workflow";

  if (!updateInfo.hasUpdate && !isInResolutionMode) {
    return null;
  }

  return (
    <>
      {isInResolutionMode ? (
        <ResolutionModeBar incompatibilities={updateInfo.incompatibilities} />
      ) : (
        <SubAgentUpdateAvailableBar
          currentVersion={updateInfo.currentVersion}
          latestVersion={updateInfo.latestVersion}
          isCompatible={updateInfo.isCompatible}
          onUpdate={handleUpdateClick}
        />
      )}
      {/* Incompatibility dialog - rendered here since this component owns the state */}
      {updateInfo.incompatibilities && (
        <IncompatibleUpdateDialog
          isOpen={showIncompatibilityDialog}
          onClose={() => setShowIncompatibilityDialog(false)}
          onConfirm={handleConfirmIncompatibleUpdate}
          currentVersion={updateInfo.currentVersion}
          latestVersion={updateInfo.latestVersion}
          agentName={beautifyString(agentName)}
          incompatibilities={updateInfo.incompatibilities}
        />
      )}
    </>
  );
}

type SubAgentUpdateAvailableBarProps = {
  currentVersion: number;
  latestVersion: number;
  isCompatible: boolean;
  onUpdate: () => void;
};

function SubAgentUpdateAvailableBar({
  currentVersion,
  latestVersion,
  isCompatible,
  onUpdate,
}: SubAgentUpdateAvailableBarProps): React.ReactElement {
  return (
    <div className="flex items-center justify-between gap-2 rounded-t-xl bg-info-surface px-3 py-2">
      <div className="flex items-center gap-2">
        <Icon icon={ArrowUp02Icon} className="h-4 w-4 text-info-foreground" />
        <Text variant="body-tight" as="span" className="text-info-foreground">
          Update available (v{currentVersion} → v{latestVersion})
        </Text>
        {!isCompatible && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Icon icon={Alert01Icon} className="h-4 w-4 text-warning" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <Text variant="small-medium-lh-normal" tone="inherit">
                Incompatible changes detected
              </Text>
              <Text variant="small-tight" tone="subtle">
                Click Update to see details
              </Text>
            </TooltipContent>
          </Tooltip>
        )}
      </div>
      <Button
        size="small"
        variant={isCompatible ? "primary" : "outline"}
        onClick={onUpdate}
        className={cn(
          "h-7",
          !isCompatible &&
            "border-warning text-warning-foreground hover:bg-warning-surface",
        )}
      >
        <Text variant="small-medium-tight" as="span" tone="inherit">
          Update
        </Text>
      </Button>
    </div>
  );
}
