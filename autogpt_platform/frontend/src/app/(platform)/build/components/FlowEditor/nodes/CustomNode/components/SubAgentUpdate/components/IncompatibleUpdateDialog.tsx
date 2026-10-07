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
          <Icon icon={Alert01Icon} className="h-5 w-5 text-yellow-500" />
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
          <Text variant="body" tone="secondary" unmask={false}>
            Updating <strong>{beautifyString(agentName)}</strong> from v
            {currentVersion} to v{latestVersion} will break some connections.
          </Text>

          {/* Input changes - two column layout */}
          {hasInputChanges && (
            <TwoColumnSection
              title="Input Changes"
              leftIcon={
                <Icon
                  icon={CancelCircleIcon}
                  className="h-4 w-4 text-red-500"
                />
              }
              leftTitle="Removed"
              leftItems={incompatibilities.missingInputs}
              rightIcon={
                <Icon
                  icon={PlusSignCircleIcon}
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
                <Icon
                  icon={CancelCircleIcon}
                  className="h-4 w-4 text-red-500"
                />
              }
              leftTitle="Removed"
              leftItems={incompatibilities.missingOutputs}
              rightIcon={
                <Icon
                  icon={PlusSignCircleIcon}
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
                <Icon
                  icon={CancelCircleIcon}
                  className="h-4 w-4 text-red-500"
                />
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
                  className="h-4 w-4 text-yellow-500"
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
              before you can save or run your agent.
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
              className="border-yellow-700 bg-yellow-600 hover:bg-yellow-700"
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
    <div className="rounded-md border border-zinc-200 p-3">
      <span className="font-medium">{title}</span>
      <div className="mt-2 grid grid-cols-2 items-start gap-4">
        {/* Left column - Breaking changes */}
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-sm text-zinc-500">
            {leftIcon}
            <span>{leftTitle}</span>
          </div>
          <ul className="mt-1.5 space-y-1">
            {leftItems.length > 0 ? (
              leftItems.map((item) => (
                <li key={item} className="text-sm text-zinc-700">
                  <code className="rounded bg-red-50 px-1 py-0.5 font-mono text-xs text-red-700">
                    {item}
                  </code>
                </li>
              ))
            ) : (
              <li className="text-sm italic text-zinc-400">None</li>
            )}
          </ul>
        </div>

        {/* Right column - Possible solutions */}
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-sm text-zinc-500">
            {rightIcon}
            <span>{rightTitle}</span>
          </div>
          <ul className="mt-1.5 space-y-1">
            {rightItems.length > 0 ? (
              rightItems.map((item) => (
                <li key={item} className="text-sm text-zinc-700">
                  <code className="rounded bg-green-50 px-1 py-0.5 font-mono text-xs text-green-700">
                    {item}
                  </code>
                </li>
              ))
            ) : (
              <li className="text-sm italic text-zinc-400">None</li>
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
    <div className="rounded-md border border-zinc-200 p-3">
      <div className="flex items-center gap-2">
        {icon}
        <span className="font-medium">{title}</span>
      </div>
      <Text variant="body" tone="muted" className="mt-1">
        {description}
      </Text>
      <ul className="mt-2 space-y-1">
        {items.map((item) => (
          <li key={item} className="ml-4 list-disc text-sm text-zinc-700">
            <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-xs">
              {item}
            </code>
          </li>
        ))}
      </ul>
    </div>
  );
}
