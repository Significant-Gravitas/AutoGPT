import React from "react";
import {
  WarningIcon,
  XCircleIcon,
  PlusCircleIcon,
} from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { beautifyString } from "@/lib/utils";
import { IncompatibilityInfo } from "@/app/(platform)/build/hooks/useSubAgentUpdate/types";

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
          <WarningIcon className="h-5 w-5 text-amber-500" weight="fill" />
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
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Updating <strong>{beautifyString(agentName)}</strong> from v
            {currentVersion} to v{latestVersion} will break some connections.
          </p>

          {/* Input changes - two column layout */}
          {hasInputChanges && (
            <TwoColumnSection
              title="Input Changes"
              leftIcon={
                <XCircleIcon className="h-4 w-4 text-red-500" weight="fill" />
              }
              leftTitle="Removed"
              leftItems={incompatibilities.missingInputs}
              rightIcon={
                <PlusCircleIcon
                  className="h-4 w-4 text-green-500"
                  weight="fill"
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
                <XCircleIcon className="h-4 w-4 text-red-500" weight="fill" />
              }
              leftTitle="Removed"
              leftItems={incompatibilities.missingOutputs}
              rightIcon={
                <PlusCircleIcon
                  className="h-4 w-4 text-green-500"
                  weight="fill"
                />
              }
              rightTitle="Added"
              rightItems={incompatibilities.newOutputs}
            />
          )}

          {hasTypeMismatches && (
            <SingleColumnSection
              icon={
                <XCircleIcon className="h-4 w-4 text-red-500" weight="fill" />
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
                <PlusCircleIcon
                  className="h-4 w-4 text-amber-500"
                  weight="fill"
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
    <div className="rounded-md border border-gray-200 p-3 dark:border-gray-700">
      <span className="font-medium">{title}</span>
      <div className="mt-2 grid grid-cols-2 items-start gap-4">
        {/* Left column - Breaking changes */}
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400">
            {leftIcon}
            <span>{leftTitle}</span>
          </div>
          <ul className="mt-1.5 space-y-1">
            {leftItems.length > 0 ? (
              leftItems.map((item) => (
                <li
                  key={item}
                  className="text-sm text-gray-700 dark:text-gray-300"
                >
                  <code className="rounded bg-red-50 px-1 py-0.5 font-mono text-xs text-red-700 dark:bg-red-900/30 dark:text-red-300">
                    {item}
                  </code>
                </li>
              ))
            ) : (
              <li className="text-sm italic text-gray-400 dark:text-gray-500">
                None
              </li>
            )}
          </ul>
        </div>

        {/* Right column - Possible solutions */}
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400">
            {rightIcon}
            <span>{rightTitle}</span>
          </div>
          <ul className="mt-1.5 space-y-1">
            {rightItems.length > 0 ? (
              rightItems.map((item) => (
                <li
                  key={item}
                  className="text-sm text-gray-700 dark:text-gray-300"
                >
                  <code className="rounded bg-green-50 px-1 py-0.5 font-mono text-xs text-green-700 dark:bg-green-900/30 dark:text-green-300">
                    {item}
                  </code>
                </li>
              ))
            ) : (
              <li className="text-sm italic text-gray-400 dark:text-gray-500">
                None
              </li>
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
    <div className="rounded-md border border-gray-200 p-3 dark:border-gray-700">
      <div className="flex items-center gap-2">
        {icon}
        <span className="font-medium">{title}</span>
      </div>
      <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
        {description}
      </p>
      <ul className="mt-2 space-y-1">
        {items.map((item) => (
          <li
            key={item}
            className="ml-4 list-disc text-sm text-gray-700 dark:text-gray-300"
          >
            <code className="rounded bg-gray-100 px-1 py-0.5 font-mono text-xs dark:bg-gray-800">
              {item}
            </code>
          </li>
        ))}
      </ul>
    </div>
  );
}
