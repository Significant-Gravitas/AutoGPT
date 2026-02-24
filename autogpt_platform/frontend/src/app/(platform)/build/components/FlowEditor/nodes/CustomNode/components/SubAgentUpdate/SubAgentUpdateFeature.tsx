import React from "react";
import { ArrowUpIcon, WarningIcon } from "@phosphor-icons/react";
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

  const agentName = nodeData.title || "Agent";

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
    <div className="flex items-center justify-between gap-2 rounded-t-xl bg-blue-50 px-3 py-2 dark:bg-blue-900/30">
      <div className="flex items-center gap-2">
        <ArrowUpIcon className="h-4 w-4 text-blue-600 dark:text-blue-400" />
        <span className="text-sm text-blue-700 dark:text-blue-300">
          Update available (v{currentVersion} → v{latestVersion})
        </span>
        {!isCompatible && (
          <Tooltip>
            <TooltipTrigger asChild>
              <WarningIcon className="h-4 w-4 text-amber-500" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              <p className="font-medium">Incompatible changes detected</p>
              <p className="text-xs text-gray-400">
                Click Update to see details
              </p>
            </TooltipContent>
          </Tooltip>
        )}
      </div>
      <Button
        size="small"
        variant={isCompatible ? "primary" : "outline"}
        onClick={onUpdate}
        className={cn(
          "h-7 text-xs",
          !isCompatible && "border-amber-500 text-amber-600 hover:bg-amber-50",
        )}
      >
        Update
      </Button>
    </div>
  );
}
