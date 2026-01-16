import React from "react";
import { Button } from "@/components/__legacy__/ui/button";
import { ArrowUp, AlertTriangle, Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { IncompatibilityInfo } from "../../../hooks/useSubAgentUpdate/types";
import { cn } from "@/lib/utils";

interface SubAgentUpdateBarProps {
  currentVersion: number;
  latestVersion: number;
  isCompatible: boolean;
  incompatibilities: IncompatibilityInfo | null;
  onUpdate: () => void;
  isInResolutionMode?: boolean;
}

export const SubAgentUpdateBar: React.FC<SubAgentUpdateBarProps> = ({
  currentVersion,
  latestVersion,
  isCompatible,
  incompatibilities,
  onUpdate,
  isInResolutionMode = false,
}) => {
  if (isInResolutionMode) {
    return <ResolutionModeBar incompatibilities={incompatibilities} />;
  }

  return (
    <div className="flex items-center justify-between gap-2 rounded-t-lg bg-blue-50 px-3 py-2 dark:bg-blue-900/30">
      <div className="flex items-center gap-2">
        <ArrowUp className="h-4 w-4 text-blue-600 dark:text-blue-400" />
        <span className="text-sm text-blue-700 dark:text-blue-300">
          Update available (v{currentVersion} → v{latestVersion})
        </span>
        {!isCompatible && (
          <Tooltip>
            <TooltipTrigger asChild>
              <AlertTriangle className="h-4 w-4 text-amber-500" />
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
        size="sm"
        variant={isCompatible ? "default" : "outline"}
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
};

interface ResolutionModeBarProps {
  incompatibilities: IncompatibilityInfo | null;
}

const ResolutionModeBar: React.FC<ResolutionModeBarProps> = ({
  incompatibilities,
}) => {
  const formatIncompatibilities = () => {
    if (!incompatibilities) return "No incompatibilities";

    const items: string[] = [];

    if (incompatibilities.missingInputs.length > 0) {
      items.push(
        `Missing inputs: ${incompatibilities.missingInputs.join(", ")}`,
      );
    }
    if (incompatibilities.missingOutputs.length > 0) {
      items.push(
        `Missing outputs: ${incompatibilities.missingOutputs.join(", ")}`,
      );
    }
    if (incompatibilities.newRequiredInputs.length > 0) {
      items.push(
        `New required inputs: ${incompatibilities.newRequiredInputs.join(", ")}`,
      );
    }
    if (incompatibilities.inputTypeMismatches.length > 0) {
      const mismatches = incompatibilities.inputTypeMismatches
        .map((m) => `${m.name} (${m.oldType} → ${m.newType})`)
        .join(", ");
      items.push(`Type changed: ${mismatches}`);
    }

    return items.join("\n");
  };

  return (
    <div className="flex items-center justify-between gap-2 rounded-t-lg bg-amber-50 px-3 py-2 dark:bg-amber-900/30">
      <div className="flex items-center gap-2">
        <AlertTriangle className="h-4 w-4 text-amber-600 dark:text-amber-400" />
        <span className="text-sm text-amber-700 dark:text-amber-300">
          Remove incompatible connections
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Info className="h-4 w-4 cursor-help text-amber-500" />
          </TooltipTrigger>
          <TooltipContent className="max-w-sm whitespace-pre-line">
            <p className="font-medium">Incompatible changes:</p>
            <p className="mt-1 text-xs">{formatIncompatibilities()}</p>
            <p className="mt-2 text-xs text-gray-400">
              Delete the red connections to continue
            </p>
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
};

export default SubAgentUpdateBar;
