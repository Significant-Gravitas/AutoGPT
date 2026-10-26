import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { beautifyString } from "@/lib/utils";
import { IncompatibilityInfo } from "@/app/(platform)/build/hooks/useSubAgentUpdate/types";
import {
  Alert01Icon,
  CancelCircleIcon,
  PlusSignCircleIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

type IncompatibleUpdateDialogProps = {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  currentVersion: number;
  latestVersion: number;
  agentName: string;
  incompatibilities: IncompatibilityInfo;
};

export function IncompatibleUpdateDialog({
  isOpen,
  onClose,
  onConfirm,
  currentVersion,
  latestVersion,
  agentName,
  incompatibilities,
}: IncompatibleUpdateDialogProps) {
  const hasMissingInputs = incompatibilities.missingInputs.length > 0;
  const hasMissingOutputs = incompatibilities.missingOutputs.length > 0;
  const hasNewInputs = incompatibilities.newInputs.length > 0;
  const hasNewOutputs = incompatibilities.newOutputs.length > 0;
  const hasNewRequired = incompatibilities.newRequiredInputs.length > 0;
  const hasTypeMismatches = incompatibilities.inputTypeMismatches.length > 0;

  const hasInputChanges = hasMissingInputs || hasNewInputs;
  const hasOutputChanges = hasMissingOutputs || hasNewOutputs;

  return (
    <Dialog
      title={
        <div className="flex items-center gap-2">
          <Icon icon={Alert01Icon} className="h-5 w-5 text-warning" />
          Incompatible Update
        </div>
      }
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
      onClose={onClose}
      styling={{ maxWidth: "32rem" }}
    >
      <Dialog.Content>
        <div className="space-y-4">
          <Text
            variant="body-tight"
            // TODO(tokens): no exact role: text-gray-600 #4B5563; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
            className="text-gray-600 dark:text-gray-400"
          >
            Updating <strong>{beautifyString(agentName)}</strong> from v
            {currentVersion} to v{latestVersion} will break some connections.
          </Text>

          {/* Input changes - two column layout */}
          {hasInputChanges && (
            <TwoColumnSection
              title="Input Changes"
              leftIcon={
                <Icon icon={CancelCircleIcon} className="h-4 w-4 text-error" />
              }
              leftTitle="Removed"
              leftItems={incompatibilities.missingInputs}
              rightIcon={
                <Icon
                  icon={PlusSignCircleIcon}
                  // TODO(tokens): no exact role: text-green-500 #16A34A
                  className="h-4 w-4 text-green-500"
                />
              }
              rightTitle="Added"
              rightItems={incompatibilities.newInputs}
            />
          )}

          {/* Output changes - two column layout */}
          {hasOutputChanges && (
            <TwoColumnSection
              title="Output Changes"
              leftIcon={
                <Icon icon={CancelCircleIcon} className="h-4 w-4 text-error" />
              }
              leftTitle="Removed"
              leftItems={incompatibilities.missingOutputs}
              rightIcon={
                <Icon
                  icon={PlusSignCircleIcon}
                  // TODO(tokens): no exact role: text-green-500 #16A34A
                  className="h-4 w-4 text-green-500"
                />
              }
              rightTitle="Added"
              rightItems={incompatibilities.newOutputs}
            />
          )}

          {hasTypeMismatches && (
            <SingleColumnSection
              icon={
                <Icon icon={CancelCircleIcon} className="h-4 w-4 text-error" />
              }
              title="Type Changed"
              description="These connected inputs have a different type:"
              items={incompatibilities.inputTypeMismatches.map(
                (m) => `${m.name} (${m.oldType} → ${m.newType})`,
              )}
            />
          )}

          {hasNewRequired && (
            <SingleColumnSection
              icon={
                <Icon
                  icon={PlusSignCircleIcon}
                  className="h-4 w-4 text-warning"
                />
              }
              title="New Required Inputs"
              description="These inputs are now required:"
              items={incompatibilities.newRequiredInputs}
            />
          )}

          <Alert variant="warning">
            <AlertDescription>
              If you proceed, you&apos;ll need to remove the broken connections
              before you can save or run your workflow.
            </AlertDescription>
          </Alert>

          <Dialog.Footer>
            <Button variant="ghost" size="small" onClick={onClose}>
              Cancel
            </Button>
            <Button
              variant="primary"
              size="small"
              onClick={onConfirm}
              // TODO(tokens): no exact role: bg-amber-600 #D97706; no fitting role for border-amber-700 #B45309; #B45309 is warning-foreground (a text colour); no fitting role for hover:bg-amber-700 #B45309; #B45309 is warning-foreground (a text colour)
              className="border-amber-700 bg-amber-600 hover:bg-amber-700"
            >
              Update Anyway
            </Button>
          </Dialog.Footer>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}

type TwoColumnSectionProps = {
  title: string;
  leftIcon: React.ReactNode;
  leftTitle: string;
  leftItems: string[];
  rightIcon: React.ReactNode;
  rightTitle: string;
  rightItems: string[];
};

function TwoColumnSection({
  title,
  leftIcon,
  leftTitle,
  leftItems,
  rightIcon,
  rightTitle,
  rightItems,
}: TwoColumnSectionProps) {
  return (
    <div
      // TODO(tokens): no fitting role for border-gray-200 #E5E7EB; #E5E7EB is sidebar-border, whose .dark value (39 39 42) differs (QUALITY 43, SECRT-3154); dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-gray-700 #374151
      className="rounded-md border border-gray-200 p-3 dark:border-gray-700"
    >
      <Text variant="large-medium-tight" as="span" tone="inherit">
        {title}
      </Text>
      <div className="mt-2 grid grid-cols-2 items-start gap-4">
        {/* Left column - Breaking changes */}
        <div className="min-w-0">
          <Text
            variant="body-tight"
            as="div"
            // TODO(tokens): no exact role: text-gray-500 #6B7280; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
            className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400"
          >
            {leftIcon}
            <span>{leftTitle}</span>
          </Text>
          <ul className="mt-1.5 space-y-1">
            {leftItems.length > 0 ? (
              leftItems.map((item) => (
                <Text
                  variant="body-tight"
                  as="li"
                  key={item}
                  // TODO(tokens): no exact role: text-gray-700 #374151; dark-mode classes kept as written (QUALITY 43): no exact role: dark:text-gray-300 #D1D5DB
                  className="text-gray-700 dark:text-gray-300"
                >
                  <Text
                    variant="small-mono-tight"
                    as="code"
                    // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no exact role: dark:bg-red-900/30 #641D1D, dark:text-red-300 #F69999
                    className="rounded bg-error-surface px-1 py-0.5 text-error-foreground dark:bg-red-900/30 dark:text-red-300"
                  >
                    {item}
                  </Text>
                </Text>
              ))
            ) : (
              <Text
                variant="body-tight-italic"
                as="li"
                // TODO(tokens): no fitting role for text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle); dark-mode classes kept as written (QUALITY 43): no exact role: dark:text-gray-500 #6B7280
                className="text-gray-400 dark:text-gray-500"
              >
                None
              </Text>
            )}
          </ul>
        </div>

        {/* Right column - Possible solutions */}
        <div className="min-w-0">
          <Text
            variant="body-tight"
            as="div"
            // TODO(tokens): no exact role: text-gray-500 #6B7280; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
            className="flex items-center gap-1.5 text-gray-500 dark:text-gray-400"
          >
            {rightIcon}
            <span>{rightTitle}</span>
          </Text>
          <ul className="mt-1.5 space-y-1">
            {rightItems.length > 0 ? (
              rightItems.map((item) => (
                <Text
                  variant="body-tight"
                  as="li"
                  key={item}
                  // TODO(tokens): no exact role: text-gray-700 #374151; dark-mode classes kept as written (QUALITY 43): no exact role: dark:text-gray-300 #D1D5DB
                  className="text-gray-700 dark:text-gray-300"
                >
                  <Text
                    variant="small-mono-tight"
                    as="code"
                    // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no exact role: dark:bg-green-900/30 #09441F, dark:text-green-300 #63C186
                    className="rounded bg-success-surface px-1 py-0.5 text-success-foreground dark:bg-green-900/30 dark:text-green-300"
                  >
                    {item}
                  </Text>
                </Text>
              ))
            ) : (
              <Text
                variant="body-tight-italic"
                as="li"
                // TODO(tokens): no fitting role for text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle); dark-mode classes kept as written (QUALITY 43): no exact role: dark:text-gray-500 #6B7280
                className="text-gray-400 dark:text-gray-500"
              >
                None
              </Text>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}

type SingleColumnSectionProps = {
  icon: React.ReactNode;
  title: string;
  description: string;
  items: string[];
};

function SingleColumnSection({
  icon,
  title,
  description,
  items,
}: SingleColumnSectionProps) {
  return (
    <div
      // TODO(tokens): no fitting role for border-gray-200 #E5E7EB; #E5E7EB is sidebar-border, whose .dark value (39 39 42) differs (QUALITY 43, SECRT-3154); dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-gray-700 #374151
      className="rounded-md border border-gray-200 p-3 dark:border-gray-700"
    >
      <div className="flex items-center gap-2">
        {icon}
        <Text variant="large-medium-tight" as="span" tone="inherit">
          {title}
        </Text>
      </div>
      <Text
        variant="body-tight"
        // TODO(tokens): no exact role: text-gray-500 #6B7280; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
        className="mt-1 text-gray-500 dark:text-gray-400"
      >
        {description}
      </Text>
      <ul className="mt-2 space-y-1">
        {items.map((item) => (
          <Text
            variant="body-tight"
            as="li"
            key={item}
            // TODO(tokens): no exact role: text-gray-700 #374151; dark-mode classes kept as written (QUALITY 43): no exact role: dark:text-gray-300 #D1D5DB
            className="ml-4 list-disc text-gray-700 dark:text-gray-300"
          >
            <Text
              variant="small-mono-tight"
              tone="inherit"
              as="code"
              // TODO(tokens): no exact role: bg-gray-100 #F3F4F6; dark-mode classes kept as written (QUALITY 43): no exact role: dark:bg-gray-800 #1F2937
              className="rounded bg-gray-100 px-1 py-0.5 dark:bg-gray-800"
            >
              {item}
            </Text>
          </Text>
        ))}
      </ul>
    </div>
  );
}
