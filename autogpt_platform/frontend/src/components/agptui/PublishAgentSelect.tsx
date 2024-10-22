import * as React from "react";
import { Button } from "./Button";
import Image from "next/image";

interface Agent {
  name: string;
  lastEdited: string;
  imageSrc: string;
}

interface PublishAgentSelectProps {
  agents: Agent[];
  onSelect: (agentName: string) => void;
  onCancel: () => void;
  onNext: () => void;
  onOpenBuilder: () => void;
}

export const PublishAgentSelect: React.FC<PublishAgentSelectProps> = ({
  agents,
  onSelect,
  onCancel,
  onNext,
  onOpenBuilder,
}) => {
  const [selectedAgent, setSelectedAgent] = React.useState<string | null>(null);

  const handleAgentClick = (agentName: string) => {
    setSelectedAgent(agentName);
    onSelect(agentName);
  };

  return (
    <div className="flex flex-col h-[749px] bg-white rounded-lg shadow border border-slate-200">
      <div className="px-6 py-6 flex-col justify-start items-center gap-3 flex">
        <h2 className="text-slate-950 text-2xl font-semibold font-['Inter'] leading-normal">
          Publish Agent
        </h2>
        <p className="text-center text-slate-500 text-sm font-normal font-['Inter'] leading-tight">
          Select your project that you'd like to publish
        </p>
      </div>
      {agents.length === 0 ? (
        <div className="flex-grow px-6 pb-6 flex-col justify-center items-center gap-4 flex">
          <p className="w-[573px] text-center text-neutral-600 text-xl font-normal font-['Geist'] leading-7">
            Uh-oh.. It seems like you don't have any agents in your library. We'd suggest you to create an agent in our builder first
          </p>
          <Button
            variant="default"
            className="px-4 py-2 bg-neutral-800 text-white hover:bg-neutral-700 rounded-[59px]"
            onClick={onOpenBuilder}
          >
            Open builder
          </Button>
        </div>
      ) : (
        <>
          <div 
            className="flex-grow px-6 pb-6 overflow-y-auto h-[calc(100%-180px)]"
            tabIndex={0}
            role="region"
            aria-label="Agent selection"
          >
            <div className="grid grid-cols-3 gap-4 pb-4">
              {agents.map((agent) => (
                <div
                  key={agent.name}
                  className={`flex flex-col cursor-pointer rounded-[10px] ${
                    selectedAgent === agent.name ? "border-2 border-gray-600" : ""
                  }`}
                  onClick={() => handleAgentClick(agent.name)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      handleAgentClick(agent.name);
                    }
                  }}
                  tabIndex={0}
                  role="button"
                  aria-pressed={selectedAgent === agent.name}
                >
                  <div className="w-full aspect-[235/158] bg-[#d9d9d9] rounded-[10px] relative overflow-hidden">
                    <Image
                      src={agent.imageSrc}
                      alt={agent.name}
                      layout="fill"
                      objectFit="cover"
                    />
                  </div>
                  <div className="mt-2 flex-col justify-start items-start gap-0.5 flex bg-white p-2 rounded-b-[10px]">
                    <div className="text-slate-900 text-sm font-medium font-['Inter'] leading-normal">
                      {agent.name}
                    </div>
                    <div className="text-slate-600 text-sm font-normal font-['Inter'] leading-normal">
                      Edited {agent.lastEdited}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div className="px-6 pb-4 pt-4 flex justify-center items-center gap-4">
            <Button
              variant="outline"
              onClick={onCancel}
              className="w-[400px] h-10 px-4 py-2 bg-white rounded-md border border-slate-200 justify-center items-center gap-2.5 inline-flex"
            >
              <span className="text-slate-950 text-sm font-medium font-['Inter'] leading-normal">
                Back
              </span>
            </Button>
            <Button
              variant="default"
              onClick={onNext}
              disabled={!selectedAgent}
              className="w-[400px] h-10 px-4 py-2 bg-slate-900 rounded-md justify-center items-center gap-2.5 inline-flex disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-800"
            >
              <span className="text-slate-50 text-sm font-medium font-['Inter'] leading-normal group-hover:text-white">
                Next
              </span>
            </Button>
          </div>
        </>
      )}
    </div>
  );
};
