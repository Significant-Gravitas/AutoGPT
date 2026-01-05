import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Button } from "@/components/atoms/Button/Button";
import { ArrowLeftIcon } from "@phosphor-icons/react";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";
import { SelectedViewLayout } from "../SelectedViewLayout";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "../../../helpers";

interface Props {
  agent: LibraryAgent;
  onClearSelectedRun: () => void;
}

export function SelectedSettingsView({ agent, onClearSelectedRun }: Props) {
  const { currentSafeMode, isPending, hasHITLBlocks, handleToggle } =
    useAgentSafeMode(agent);

  return (
    <SelectedViewLayout agent={agent} onSelectSettings={() => {}}>
      <div className="flex flex-col gap-4">
        <div
          className={`${AGENT_LIBRARY_SECTION_PADDING_X} mb-8 flex items-center gap-3`}
        >
          <Button
            variant="icon"
            size="small"
            onClick={onClearSelectedRun}
            className="w-[2.375rem]"
          >
            <ArrowLeftIcon />
          </Button>
          <Text variant="h2">Agent Settings</Text>
        </div>

        <div className={AGENT_LIBRARY_SECTION_PADDING_X}>
          {!hasHITLBlocks ? (
            <div className="rounded-xl border border-zinc-100 bg-white p-6">
              <Text variant="body" className="text-muted-foreground">
                This agent doesn&apos;t have any human-in-the-loop blocks, so
                there are no settings to configure.
              </Text>
            </div>
          ) : (
            <div className="flex w-full max-w-2xl flex-col items-start gap-4 rounded-xl border border-zinc-100 bg-white p-6">
              <div className="flex w-full items-start justify-between gap-4">
                <div className="flex-1">
                  <Text variant="large-semibold">Require human approval</Text>
                  <Text variant="large" className="mt-1 text-zinc-900">
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
    </SelectedViewLayout>
  );
}
