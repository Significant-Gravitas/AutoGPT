import React from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { IncompatibilityInfo } from "@/app/(platform)/build/hooks/useSubAgentUpdate/types";
import { Alert01Icon, InformationCircleIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

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
              <span className="text-zinc-400">
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
    <div className="flex items-center justify-between gap-2 rounded-t-xl bg-yellow-50 px-3 py-2">
      <div className="flex items-center gap-2">
        <Icon icon={Alert01Icon} className="h-4 w-4 text-yellow-600" />
        <span className="text-sm text-yellow-700">
          Remove incompatible connections
        </span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Icon
              icon={InformationCircleIcon}
              className="h-4 w-4 cursor-help text-yellow-500"
            />
          </TooltipTrigger>
          <TooltipContent className="max-w-sm">
            <Text variant="small" tone="primary" className="mb-2 font-semibold">
              Incompatible changes:
            </Text>
            <div className="text-xs">{renderIncompatibilities()}</div>
            <Text variant="small" className="mt-2 text-zinc-400">
              {(incompatibilities?.newRequiredInputs.length ?? 0) > 0
                ? "Replace / delete"
                : "Delete"}{" "}
              the red connections to continue
            </Text>
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}
