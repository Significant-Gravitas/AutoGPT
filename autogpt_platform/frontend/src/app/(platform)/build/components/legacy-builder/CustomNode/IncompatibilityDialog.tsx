import React from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { Button } from "@/components/__legacy__/ui/button";
import { AlertTriangle, XCircle, PlusCircle } from "lucide-react";
import { IncompatibilityInfo } from "../../../hooks/useSubAgentUpdate/types";
import { beautifyString } from "@/lib/utils";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";

interface IncompatibilityDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  currentVersion: number;
  latestVersion: number;
  agentName: string;
  incompatibilities: IncompatibilityInfo;
}

export const IncompatibilityDialog: React.FC<IncompatibilityDialogProps> = ({
  isOpen,
  onClose,
  onConfirm,
  currentVersion,
  latestVersion,
  agentName,
  incompatibilities,
}) => {
  const hasMissingInputs = incompatibilities.missingInputs.length > 0;
  const hasMissingOutputs = incompatibilities.missingOutputs.length > 0;
  const hasNewInputs = incompatibilities.newInputs.length > 0;
  const hasNewOutputs = incompatibilities.newOutputs.length > 0;
  const hasNewRequired = incompatibilities.newRequiredInputs.length > 0;
  const hasTypeMismatches = incompatibilities.inputTypeMismatches.length > 0;

  const hasInputChanges = hasMissingInputs || hasNewInputs;
  const hasOutputChanges = hasMissingOutputs || hasNewOutputs;

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-500" />
            Incompatible Update
          </DialogTitle>
          <DialogDescription>
            Updating <strong>{beautifyString(agentName)}</strong> from v
            {currentVersion} to v{latestVersion} will break some connections.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-2">
          {/* Input changes - two column layout */}
          {hasInputChanges && (
            <TwoColumnSection
              title="Input Changes"
              leftIcon={<XCircle className="h-4 w-4 text-red-500" />}
              leftTitle="Removed"
              leftItems={incompatibilities.missingInputs}
              rightIcon={<PlusCircle className="h-4 w-4 text-green-500" />}
              rightTitle="Added"
              rightItems={incompatibilities.newInputs}
            />
          )}

          {/* Output changes - two column layout */}
          {hasOutputChanges && (
            <TwoColumnSection
              title="Output Changes"
              leftIcon={<XCircle className="h-4 w-4 text-red-500" />}
              leftTitle="Removed"
              leftItems={incompatibilities.missingOutputs}
              rightIcon={<PlusCircle className="h-4 w-4 text-green-500" />}
              rightTitle="Added"
              rightItems={incompatibilities.newOutputs}
            />
          )}

          {hasTypeMismatches && (
            <SingleColumnSection
              icon={<XCircle className="h-4 w-4 text-red-500" />}
              title="Type Changed"
              description="These connected inputs have a different type:"
              items={incompatibilities.inputTypeMismatches.map(
                (m) => `${m.name} (${m.oldType} → ${m.newType})`,
              )}
            />
          )}

          {hasNewRequired && (
            <SingleColumnSection
              icon={<PlusCircle className="h-4 w-4 text-amber-500" />}
              title="New Required Inputs"
              description="These inputs are now required:"
              items={incompatibilities.newRequiredInputs}
            />
          )}
        </div>

        <Alert variant="warning">
          <AlertDescription>
            If you proceed, you&apos;ll need to remove the broken connections
            before you can save or run your agent.
          </AlertDescription>
        </Alert>

        <DialogFooter className="gap-2 sm:gap-0">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirm}
            className="bg-amber-600 hover:bg-amber-700"
          >
            Update Anyway
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

interface TwoColumnSectionProps {
  title: string;
  leftIcon: React.ReactNode;
  leftTitle: string;
  leftItems: string[];
  rightIcon: React.ReactNode;
  rightTitle: string;
  rightItems: string[];
}

const TwoColumnSection: React.FC<TwoColumnSectionProps> = ({
  title,
  leftIcon,
  leftTitle,
  leftItems,
  rightIcon,
  rightTitle,
  rightItems,
}) => (
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

interface SingleColumnSectionProps {
  icon: React.ReactNode;
  title: string;
  description: string;
  items: string[];
}

const SingleColumnSection: React.FC<SingleColumnSectionProps> = ({
  icon,
  title,
  description,
  items,
}) => (
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

export default IncompatibilityDialog;
