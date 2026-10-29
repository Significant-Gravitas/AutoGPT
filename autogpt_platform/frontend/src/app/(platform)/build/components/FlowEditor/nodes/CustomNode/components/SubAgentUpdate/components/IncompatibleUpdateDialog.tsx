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
          <Text variant="body-tight" tone="secondary">
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
                  className="h-4 w-4 text-success"
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
                  className="h-4 w-4 text-success"
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
              className="border-warning-foreground bg-warning-foreground hover:bg-warning-foreground/90"
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
    <div className="rounded-md border border-border-base p-3">
      <Text variant="large-medium-tight" as="span" tone="inherit">
        {title}
      </Text>
      <div className="mt-2 grid grid-cols-2 items-start gap-4">
        {/* Left column - Breaking changes */}
        <div className="min-w-0">
          <Text
            variant="body-tight"
            tone="muted"
            as="div"
            className="flex items-center gap-1.5"
          >
            {leftIcon}
            <span>{leftTitle}</span>
          </Text>
          <ul className="mt-1.5 space-y-1">
            {leftItems.length > 0 ? (
              leftItems.map((item) => (
                <Text variant="body-tight" tone="body" as="li" key={item}>
                  <Text
                    variant="small-mono-tight"
                    as="code"
                    className="rounded bg-error-surface px-1 py-0.5 text-error-foreground"
                  >
                    {item}
                  </Text>
                </Text>
              ))
            ) : (
              <Text variant="body-tight-italic" tone="subtle" as="li">
                None
              </Text>
            )}
          </ul>
        </div>

        {/* Right column - Possible solutions */}
        <div className="min-w-0">
          <Text
            variant="body-tight"
            tone="muted"
            as="div"
            className="flex items-center gap-1.5"
          >
            {rightIcon}
            <span>{rightTitle}</span>
          </Text>
          <ul className="mt-1.5 space-y-1">
            {rightItems.length > 0 ? (
              rightItems.map((item) => (
                <Text variant="body-tight" tone="body" as="li" key={item}>
                  <Text
                    variant="small-mono-tight"
                    as="code"
                    className="rounded bg-success-surface px-1 py-0.5 text-success-foreground"
                  >
                    {item}
                  </Text>
                </Text>
              ))
            ) : (
              <Text variant="body-tight-italic" tone="subtle" as="li">
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
    <div className="rounded-md border border-border-base p-3">
      <div className="flex items-center gap-2">
        {icon}
        <Text variant="large-medium-tight" as="span" tone="inherit">
          {title}
        </Text>
      </div>
      <Text variant="body-tight" tone="muted" className="mt-1">
        {description}
      </Text>
      <ul className="mt-2 space-y-1">
        {items.map((item) => (
          <Text
            variant="body-tight"
            tone="body"
            as="li"
            key={item}
            className="ml-4 list-disc"
          >
            <Text
              variant="small-mono-tight"
              tone="inherit"
              as="code"
              className="rounded bg-surface-hover px-1 py-0.5"
            >
              {item}
            </Text>
          </Text>
        ))}
      </ul>
    </div>
  );
}
