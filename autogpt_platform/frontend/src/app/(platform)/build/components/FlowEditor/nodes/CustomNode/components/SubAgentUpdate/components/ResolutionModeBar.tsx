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
          <Text variant="small-semibold-tight" as="span" tone="inherit">
            Missing inputs:{" "}
          </Text>
          {incompatibilities.missingInputs.map((name, i) => (
            <React.Fragment key={name}>
              <Text variant="small-mono-tight" as="code" tone="inherit">
                {name}
              </Text>
              {i < incompatibilities.missingInputs.length - 1 && ", "}
            </React.Fragment>
          ))}
        </div>,
      );
    }
    if (incompatibilities.missingOutputs.length > 0) {
      sections.push(
        <div key="missing-outputs" className="mb-1">
          <Text variant="small-semibold-tight" as="span" tone="inherit">
            Missing outputs:{" "}
          </Text>
          {incompatibilities.missingOutputs.map((name, i) => (
            <React.Fragment key={name}>
              <Text variant="small-mono-tight" as="code" tone="inherit">
                {name}
              </Text>
              {i < incompatibilities.missingOutputs.length - 1 && ", "}
            </React.Fragment>
          ))}
        </div>,
      );
    }
    if (incompatibilities.newRequiredInputs.length > 0) {
      sections.push(
        <div key="new-required" className="mb-1">
          <Text variant="small-semibold-tight" as="span" tone="inherit">
            New required inputs:{" "}
          </Text>
          {incompatibilities.newRequiredInputs.map((name, i) => (
            <React.Fragment key={name}>
              <Text variant="small-mono-tight" as="code" tone="inherit">
                {name}
              </Text>
              {i < incompatibilities.newRequiredInputs.length - 1 && ", "}
            </React.Fragment>
          ))}
        </div>,
      );
    }
    if (incompatibilities.inputTypeMismatches.length > 0) {
      sections.push(
        <div key="type-mismatches" className="mb-1">
          <Text variant="small-semibold-tight" as="span" tone="inherit">
            Type changed:{" "}
          </Text>
          {incompatibilities.inputTypeMismatches.map((m, i) => (
            <React.Fragment key={m.name}>
              <Text variant="small-mono-tight" as="code" tone="inherit">
                {m.name}
              </Text>
              <span
                // TODO(tokens): no fitting role for text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
                className="text-gray-400"
              >
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
    <div
      // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no exact role: dark:bg-amber-900/30 #78350F
      className="flex items-center justify-between gap-2 rounded-t-xl bg-warning-surface px-3 py-2 dark:bg-amber-900/30"
    >
      <div className="flex items-center gap-2">
        <Icon
          icon={Alert01Icon}
          // TODO(tokens): no exact role: text-amber-600 #D97706; dark-mode classes kept as written (QUALITY 43): no exact role: dark:text-amber-400 #FBBF24
          className="h-4 w-4 text-amber-600 dark:text-amber-400"
        />
        <Text
          // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no exact role: dark:text-amber-300 #FCD34D
          className="dark:text-amber-300"
          variant="body-tight"
          tone="warning"
          as="span"
        >
          Remove incompatible connections
        </Text>
        <Tooltip>
          <TooltipTrigger asChild>
            <Icon
              icon={InformationCircleIcon}
              className="h-4 w-4 cursor-help text-warning"
            />
          </TooltipTrigger>
          <TooltipContent className="max-w-sm">
            <Text
              variant="small-semibold-lh-normal"
              tone="inherit"
              className="mb-2"
            >
              Incompatible changes:
            </Text>
            <Text variant="small-tight" tone="inherit" as="div">
              {renderIncompatibilities()}
            </Text>
            <Text
              variant="small-tight"
              // TODO(tokens): no fitting role for text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
              className="mt-2 text-gray-400"
            >
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
