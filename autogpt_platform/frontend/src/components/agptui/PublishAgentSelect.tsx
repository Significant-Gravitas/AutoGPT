"use client";

import * as React from "react";
import Image from "next/image";
import { Button } from "../agptui/Button";
import { X } from "lucide-react";

export interface Agent {
  name: string;
  id: string;
  version: number;
  lastEdited: string;
  imageSrc: string;
}

interface PublishAgentSelectProps {
  agents: Agent[];
  onSelect: (agentId: string, agentVersion: number) => void;
  onCancel: () => void;
  onNext: (agentId: string, agentVersion: number) => void;
  onClose: () => void;
  onOpenBuilder: () => void;
}

export const PublishAgentSelect: React.FC<PublishAgentSelectProps> = ({
  agents,
  onSelect,
  onCancel,
  onNext,
  onClose,
  onOpenBuilder,
}) => {
  const [selectedAgentId, setSelectedAgentId] = React.useState<string | null>(
    null,
  );
  const [selectedAgentVersion, setSelectedAgentVersion] = React.useState<
    number | null
  >(null);

  const handleAgentClick = (
    agentName: string,
    agentId: string,
    agentVersion: number,
  ) => {
    setSelectedAgentId(agentId);
    setSelectedAgentVersion(agentVersion);
    onSelect(agentId, agentVersion);
  };

  return (
    <div className="m-auto flex h-fit w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg dark:bg-gray-800">
      {/* Top */}
      <div className="relative flex h-28 items-center justify-center border-b border-slate-200 dark:border-slate-700">
        <div className="absolute right-4 top-4">
          <Button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-transparent p-0 transition-colors hover:bg-gray-200"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
        <div className="text-center">
          <h3 className="font-poppins text-2xl font-semibold text-neutral-900">
            Publish Agent
          </h3>
          <p className="font-sans text-base font-normal text-neutral-600">
            Select your project that you&apos;d like to publish
          </p>
        </div>
      </div>

      {agents.length === 0 ? (
        <div className="inline-flex h-96 flex-col items-center justify-center gap-[29px] px-4 py-5 sm:px-6">
          <div className="w-full text-center font-sans text-lg font-normal leading-7 text-neutral-600 dark:text-neutral-400 sm:w-[573px] sm:text-xl">
            Uh-oh.. It seems like you don&apos;t have any agents in your
            library.
            <br />
            We&apos;d suggest you to create an agent in our builder first
          </div>
          <Button
            onClick={onOpenBuilder}
            size="lg"
            className="bg-neutral-800 text-white hover:bg-neutral-900"
          >
            Open builder
          </Button>
        </div>
      ) : (
        <>
          <div className="flex-grow overflow-hidden">
            <h3 className="sr-only">List of agents</h3>
            <div
              className="h-72 overflow-y-auto px-6 py-6 sm:h-[400px] md:h-[500px]"
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
                      className={`cursor-pointer overflow-hidden rounded-2xl transition-all ${
                        selectedAgentId === agent.id
                          ? "shadow-lg ring-4 ring-violet-600"
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
                      <div className="relative h-32 bg-gray-100 dark:bg-gray-700 sm:h-40">
                        <Image
                          src={agent.imageSrc}
                          alt={agent.name}
                          fill
                          style={{ objectFit: "cover" }}
                        />
                      </div>
                      <div className="p-3">
                        <p className="font-poppins text-base font-medium leading-normal text-neutral-800 dark:text-neutral-100 sm:text-base">
                          {agent.name}
                        </p>
                        <small className="font-sans text-xs font-normal leading-[14px] text-neutral-500 dark:text-neutral-400 sm:text-sm">
                          Edited {agent.lastEdited}
                        </small>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="flex justify-between gap-4 border-t border-slate-200 p-4 dark:border-slate-700 sm:p-6">
            <Button
              onClick={onCancel}
              size="lg"
              className="flex w-full items-center justify-center sm:flex-1"
            >
              Back
            </Button>
            <Button
              onClick={() => {
                if (selectedAgentId && selectedAgentVersion) {
                  onNext(selectedAgentId, selectedAgentVersion);
                }
              }}
              disabled={!selectedAgentId || !selectedAgentVersion}
              size="lg"
              className="flex w-full items-center justify-center bg-neutral-800 text-white hover:bg-neutral-900 sm:flex-1"
            >
              Next
            </Button>
          </div>
        </>
      )}
    </div>
  );
};
