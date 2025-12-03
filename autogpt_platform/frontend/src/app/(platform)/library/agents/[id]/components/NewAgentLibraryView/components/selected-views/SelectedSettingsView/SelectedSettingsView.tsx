import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Breadcrumbs } from "@/components/molecules/Breadcrumbs/Breadcrumbs";
import { Button } from "@/components/atoms/Button/Button";
import { ArrowLeftIcon } from "@phosphor-icons/react";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";

interface SelectedSettingsViewProps {
  agent: LibraryAgent;
  onClearSelectedRun: () => void;
}

export function SelectedSettingsView({
  agent,
  onClearSelectedRun,
}: SelectedSettingsViewProps) {
  const { currentSafeMode, isPending, hasHITLBlocks, handleToggle } =
    useAgentSafeMode(agent);

  return (
    <div className="flex h-full flex-col gap-4">
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          size="small"
          onClick={onClearSelectedRun}
          leftIcon={<ArrowLeftIcon size={20} />}
        >
          Back
        </Button>
        <Text variant="h1">Agent Settings</Text>
      </div>

      <div className="flex-1">
        {!hasHITLBlocks ? (
          <div className="rounded-lg border p-6">
            <Text variant="body" className="text-muted-foreground">
              This agent doesn't have any human-in-the-loop blocks, so there are
              no settings to configure.
            </Text>
          </div>
        ) : (
          <div className="rounded-lg border p-6">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1">
                <Text
                  variant="body-medium"
                  className="font-semibold text-black"
                >
                  Require human approval
                </Text>
                <Text variant="body" className="mt-1 text-gray-600">
                  The agent will pause and wait for your review before
                  continuing
                </Text>
              </div>
              <Switch
                checked={currentSafeMode || false}
                onCheckedChange={handleToggle}
                disabled={isPending}
                className="mt-1"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
