import React from "react";
import { InfoIcon, WarningIcon } from "@phosphor-icons/react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { IncompatibilityInfo } from "@/app/(platform)/build/hooks/useSubAgentUpdate/types";

type ResolutionModeBarProps = {
  incompatibilities: IncompatibilityInfo | null;
};

export function ResolutionModeBar({
  incompatibilities,
}: ResolutionModeBarProps): React.ReactElement {
  const renderIncompatibilities = () => {
    if (!incompatibilities) return <span>No incompatibilities</span>;

    const sections: React.ReactNode[] = [];

    if (incompatibilities.missingInputs.length > 0) {
      sections.push(
        <div key="missing-inputs" className="mb-1">
          <span className="font-semibold">Missing inputs: </span>
          {incompatibilities.missingInputs.map((name, i) => (
            <React.Fragment key={name}>
              <code className="font-mono">{name}</code>
              {i < incompatibilities.missingInputs.length - 1 && ", "}
            </React.Fragment>
          ))}
        </div>,
      );
    }
    if (incompatibilities.missingOutputs.length > 0) {
      sections.push(
        <div key="missing-outputs" className="mb-1">
          <span className="font-semibold">Missing outputs: </span>
          {incompatibilities.missingOutputs.map((name, i) => (
            <React.Fragment key={name}>
              <code className="font-mono">{name}</code>
              {i < incompatibilities.missingOutputs.length - 1 && ", "}
            </React.Fragment>
          ))}
        </div>,
      );
    }
    if (incompatibilities.newRequiredInputs.length > 0) {
      sections.push(
        <div key="new-required" className="mb-1">
          <span className="font-semibold">New required inputs: </span>
          {incompatibilities.newRequiredInputs.map((name, i) => (
            <React.Fragment key={name}>
              <code className="font-mono">{name}</code>
              {i < incompatibilities.newRequiredInputs.length - 1 && ", "}
            </React.Fragment>
          ))}
        </div>,
      );
    }
    if (incompatibilities.inputTypeMismatches.length > 0) {
      sections.push(
        <div key="type-mismatches" className="mb-1">
          <span className="font-semibold">Type changed: </span>
          {incompatibilities.inputTypeMismatches.map((m, i) => (
            <React.Fragment key={m.name}>
              <code className="font-mono">{m.name}</code>
              <span className="text-gray-400">
                {" "}
                ({m.oldType} → {m.newType})
              </span>
              {i < incompatibilities.inputTypeMismatches.length - 1 && ", "}
            </React.Fragment>
          ))}
        </div>,
      );
    }

    return <>{sections}</>;
  };

  return (
    <div className="flex items-center justify-between gap-2 rounded-t-xl bg-amber-50 px-3 py-2 dark:bg-amber-900/30">
      <div className="flex items-center gap-2">
        <WarningIcon className="h-4 w-4 text-amber-600 dark:text-amber-400" />
        <span className="text-sm text-amber-700 dark:text-amber-300">
          Remove incompatible connections
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <InfoIcon className="h-4 w-4 cursor-help text-amber-500" />
          </TooltipTrigger>
          <TooltipContent className="max-w-sm">
            <p className="mb-2 font-semibold">Incompatible changes:</p>
            <div className="text-xs">{renderIncompatibilities()}</div>
            <p className="mt-2 text-xs text-gray-400">
              {(incompatibilities?.newRequiredInputs.length ?? 0) > 0
                ? "Replace / delete"
                : "Delete"}{" "}
              the red connections to continue
            </p>
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}
