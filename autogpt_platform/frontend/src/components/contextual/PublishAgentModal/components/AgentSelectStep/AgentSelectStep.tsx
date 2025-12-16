"use client";

import * as React from "react";
import Image from "next/image";
import { Text } from "../../../../atoms/Text/Text";
import { Button } from "../../../../atoms/Button/Button";
import { StepHeader } from "../StepHeader";
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
    publishedSubmissionData?: any,
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
    agents,
    isLoading,
    error,
    // State
    selectedAgentId,
    // Handlers
    handleAgentClick,
    handleNext,
    // Utils
    getPublishedVersion,
    // Computed
    isNextDisabled,
  } = useAgentSelectStep({ onSelect, onNext });

  if (isLoading) {
    return (
      <div className="mx-auto flex min-h-[70vh] w-full flex-col">
        <StepHeader
          title="Publish Agent"
          description="Select your project that you'd like to publish"
        />
        <div className="flex-grow p-4 sm:p-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="overflow-hidden rounded-2xl border border-neutral-200"
              >
                <Skeleton className="h-32 w-full sm:h-40" />
                <div className="flex flex-col gap-2 p-3">
                  <Skeleton className="h-5 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
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
      <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl">
        <StepHeader
          title="Publish Agent"
          description="Select your project that you'd like to publish"
        />
        <div className="inline-flex h-[370px] flex-col items-center justify-center gap-[29px] px-4 py-5 sm:px-6">
          <Text variant="lead" className="text-center text-red-600">
            Failed to load agents. Please try again.
          </Text>
          <Button onClick={() => window.location.reload()} variant="secondary">
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl">
      <StepHeader
        title="Publish Agent"
        description="Select your project that you'd like to publish"
      />

      {agents.length === 0 ? (
        <div className="inline-flex h-[370px] flex-col items-center justify-center gap-[29px] px-4 py-5 sm:px-6">
          <Text variant="lead" className="text-center">
            Uh-oh.. It seems like you don&apos;t have any agents in your
            library. We&apos;d suggest you to create an agent in our builder
            first
          </Text>
          <Button
            onClick={onOpenBuilder}
            className="bg-neutral-800 text-white hover:bg-neutral-900"
          >
            Open builder
          </Button>
        </div>
      ) : (
        <>
          <div className="flex-grow overflow-hidden p-4 sm:p-6">
            <h3 className="sr-only">List of agents</h3>
            <div
              className={cn(
                scrollbarStyles,
                "h-[300px] overflow-y-auto pr-2 sm:h-[400px] md:h-[500px]",
              )}
              role="region"
              aria-labelledby="agentListHeading"
            >
              <div id="agentListHeading" className="sr-only">
                Scrollable list of agents
              </div>
              <div className="p-2">
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                  {agents.map((agent) => (
                    <div
                      key={agent.id}
                      data-testid="agent-card"
                      className={`cursor-pointer select-none overflow-hidden rounded-2xl border border-neutral-200 shadow-sm transition-all ${
                        selectedAgentId === agent.id
                          ? "border-transparent shadow-none ring-4 ring-violet-600"
                          : "hover:shadow-md"
                      }`}
                      onClick={() =>
                        handleAgentClick(agent.name, agent.id, agent.version)
                      }
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          handleAgentClick(agent.name, agent.id, agent.version);
                        }
                      }}
                      tabIndex={0}
                      role="button"
                      aria-pressed={selectedAgentId === agent.id}
                    >
                      <div className="relative h-32 bg-zinc-400 sm:h-40">
                        <Image
                          src={agent.imageSrc}
                          alt={agent.name}
                          className="object-cover"
                          fill
                          sizes="(max-width: 640px) 100vw, (max-width: 768px) 50vw, 33vw"
                        />
                      </div>
                      <div className="flex flex-col gap-2 p-3">
                        <Text variant="large-medium" className="line-clamp-2">
                          {agent.name}
                        </Text>
                        <div className="flex items-center justify-between gap-2">
                          <div className="flex-1">
                            <Text variant="small" className="!text-neutral-500">
                              Edited {agent.lastEdited}
                            </Text>
                            {agent.isMarketplaceUpdate &&
                              (() => {
                                const publishedVersion = getPublishedVersion(
                                  agent.id,
                                );
                                return (
                                  publishedVersion && (
                                    <Text
                                      variant="small"
                                      className="block text-xs !text-neutral-500"
                                    >
                                      v{publishedVersion} → v{agent.version}
                                    </Text>
                                  )
                                );
                              })()}
                          </div>
                          {agent.isMarketplaceUpdate && (
                            <span className="shrink-0 rounded-full bg-blue-100 px-2 py-1 text-xs font-medium text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                              Update
                            </span>
                          )}
                          {!agent.isMarketplaceUpdate && (
                            <span className="shrink-0 rounded-full bg-green-100 px-2 py-1 text-xs font-medium text-green-800 dark:bg-green-900 dark:text-green-200">
                              New
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-between gap-4 p-4 sm:p-6">
            <Button
              variant="secondary"
              onClick={onCancel}
              className="w-full sm:flex-1"
            >
              Back
            </Button>
            <Button
              onClick={handleNext}
              disabled={isNextDisabled}
              className="w-full bg-neutral-800 text-white hover:bg-neutral-900 sm:flex-1"
            >
              Next
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
