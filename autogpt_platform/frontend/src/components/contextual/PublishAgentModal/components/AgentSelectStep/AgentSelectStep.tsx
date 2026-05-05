"use client";

import {
  CheckCircleIcon,
  PlusIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";

import { Text } from "../../../../atoms/Text/Text";
import { Button } from "../../../../atoms/Button/Button";
import { StepHeader } from "../StepHeader";
import { StepFooter } from "../StepFooter";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { useAgentSelectStep } from "./useAgentSelectStep";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";

interface Props {
  onSelect: (agentId: string, agentVersion: number) => void;
  onCancel: () => void;
  onNext: (
    agentId: string,
    agentVersion: number,
    agentData: {
      name: string;
      description: string;
      imageSrc: string;
      recommendedScheduleCron: string | null;
    },
  ) => void;
  onOpenBuilder: () => void;
}

export function AgentSelectStep({
  onSelect,
  onCancel,
  onNext,
  onOpenBuilder,
}: Props) {
  const {
    // Data
    myAgents,
    isLoading,
    error,
    // State
    selectedAgentId,
    // Handlers
    handleAgentClick,
    handleNext,
    // Computed
    isNextDisabled,
  } = useAgentSelectStep({ onSelect, onNext });

  if (isLoading) {
    return (
      <div className="mx-auto flex w-full flex-col">
        <StepHeader
          title="Choose an agent"
          description="Pick the saved agent version you want to send to marketplace review."
          currentStep="select"
        />
        <div className="flex-grow pb-5">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="flex items-center gap-3 rounded-[12px] border border-zinc-200 bg-white p-3"
              >
                <div className="flex flex-1 flex-col gap-2">
                  <Skeleton className="h-4 w-1/3" />
                  <Skeleton className="h-3 w-2/3" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto flex w-full flex-col">
        <StepHeader
          title="Choose an agent"
          description="Pick the saved agent version you want to send to marketplace review."
          currentStep="select"
        />
        <div className="mt-5 flex min-h-[320px] flex-col items-center justify-center gap-4 rounded-[18px] border border-rose-100 bg-rose-50 px-6 py-8 text-center">
          <WarningCircleIcon
            size={32}
            weight="duotone"
            className="text-rose-600"
          />
          <Text variant="large-medium" className="text-rose-900">
            We could not load your agents
          </Text>
          <Text variant="body" className="max-w-[420px] text-rose-700">
            Refresh the list and try again. Your current marketplace submissions
            are unchanged.
          </Text>
          <Button onClick={() => window.location.reload()} variant="secondary">
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full flex-col">
      <StepHeader
        title="Choose an agent"
        description="Pick the saved agent version you want to send to marketplace review."
        currentStep="select"
      />

      {myAgents.length === 0 ? (
        <div className="mt-5 flex min-h-[320px] flex-col items-center justify-center gap-4 rounded-[18px] border border-dashed border-zinc-300 bg-zinc-50 px-6 py-8 text-center">
          <div className="flex size-11 items-center justify-center rounded-full bg-white text-zinc-700 shadow-[0_1px_2px_rgba(15,15,20,0.06)]">
            <PlusIcon size={20} weight="bold" />
          </div>
          <Text variant="large-medium" className="text-textBlack">
            No publishable agents yet
          </Text>
          <Text variant="body" className="max-w-[460px] text-zinc-600">
            Create and save an agent in the builder. It will appear here when a
            version is ready to submit.
          </Text>
          <Button onClick={onOpenBuilder}>Open builder</Button>
        </div>
      ) : (
        <>
          <div className="flex-grow overflow-hidden pb-5">
            <h3 className="sr-only">List of agents</h3>
            <div
              className={cn(
                scrollbarStyles,
                "max-h-[48vh] min-h-[280px] overflow-y-auto pr-2",
              )}
              role="region"
              aria-labelledby="agentListHeading"
            >
              <div id="agentListHeading" className="sr-only">
                Scrollable list of agents
              </div>
              <div className="grid grid-cols-1 gap-2 p-1 sm:grid-cols-2">
                {myAgents.map((agent) => {
                  const isSelected = selectedAgentId === agent.id;
                  return (
                    <button
                      type="button"
                      key={agent.id}
                      data-testid="agent-card"
                      className={cn(
                        "group flex w-full cursor-pointer select-none items-center gap-3 rounded-[12px] border bg-white p-3 text-left transition-[border-color,box-shadow] duration-150 hover:border-purple-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 focus-visible:ring-offset-2",
                        isSelected
                          ? "border-purple-500 bg-purple-50/40 shadow-[0_0_0_3px_rgba(119,51,245,0.12)]"
                          : "border-zinc-200",
                      )}
                      onClick={() =>
                        handleAgentClick(agent.name, agent.id, agent.version)
                      }
                      aria-pressed={isSelected}
                    >
                      <div className="flex min-w-0 flex-1 flex-col gap-1">
                        <div className="flex min-w-0 items-center gap-2">
                          <Text
                            variant="body-medium"
                            as="span"
                            className="truncate text-textBlack"
                          >
                            {agent.name}
                          </Text>
                          <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] font-medium text-zinc-600">
                            v{agent.version}
                          </span>
                        </div>
                        <Text
                          variant="small"
                          as="span"
                          className="truncate text-zinc-500"
                        >
                          {agent.description
                            ? agent.description
                            : `Edited ${agent.lastEdited}`}
                        </Text>
                      </div>
                      {isSelected ? (
                        <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-purple-500 text-white">
                          <CheckCircleIcon size={14} weight="fill" />
                        </span>
                      ) : (
                        <span
                          aria-hidden
                          className="size-5 shrink-0 rounded-full border border-zinc-300"
                        />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <StepFooter
            secondary={
              <Button
                variant="secondary"
                size="small"
                onClick={onCancel}
                className="w-full sm:w-auto"
              >
                Cancel
              </Button>
            }
            primary={
              <Button
                size="small"
                onClick={handleNext}
                disabled={isNextDisabled}
                className="w-full sm:w-auto"
              >
                Continue
              </Button>
            }
          />
        </>
      )}
    </div>
  );
}
