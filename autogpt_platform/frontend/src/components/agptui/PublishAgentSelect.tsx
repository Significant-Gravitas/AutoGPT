import * as React from "react";
import Image from "next/image";
import { Button } from "../agptui/Button";

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
  const [selectedAgent, setSelectedAgent] = React.useState<string | null>(null);

  const handleAgentClick = (agentName: string) => {
    setSelectedAgent(agentName);
    onSelect(agentName);
  };

  return (
    <div className="w-full max-w-[900px] bg-white rounded-3xl shadow-lg flex flex-col mx-auto">
      <div className="p-4 sm:p-6 border-b border-slate-200 relative">
        <div className="absolute top-4 right-4">
          <button
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-gray-100 flex items-center justify-center hover:bg-gray-200 transition-colors"
            aria-label="Close"
          >
            <svg
              width="14"
              height="14"
              viewBox="0 0 14 14"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M1 1L13 13M1 13L13 1"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        </div>
        <h2 className="text-neutral-900 text-xl sm:text-2xl font-semibold font-['Poppins'] leading-loose text-center mb-2">Publish Agent</h2>
        <p className="text-neutral-600 text-sm sm:text-base font-normal font-['Geist'] leading-7 text-center">Select your project that you'd like to publish</p>
      </div>
      
      {agents.length === 0 ? (
        <div className="h-[370px] px-4 sm:px-6 py-5 flex-col justify-center items-center gap-[29px] inline-flex">
          <div className="w-full sm:w-[573px] text-center text-neutral-600 text-lg sm:text-xl font-normal font-['Geist'] leading-7">
            Uh-oh.. It seems like you don't have any agents in your library.
            <br />
            We'd suggest you to create an agent in our builder first
          </div>
          <Button
            onClick={onOpenBuilder}
            variant="default"
            size="lg"
            className="text-white bg-neutral-800 hover:bg-neutral-900"
          >
            Open builder
          </Button>
        </div>
      ) : (
        <>
          <div className="flex-grow p-4 sm:p-6 overflow-hidden">
            <h3 className="sr-only">List of agents</h3>
            <div 
              className="h-[300px] sm:h-[400px] md:h-[500px] overflow-y-auto pr-2"
              role="region"
              aria-labelledby="agentListHeading"
            >
              <div id="agentListHeading" className="sr-only">Scrollable list of agents</div>
              <div className="p-2">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {agents.map((agent) => (
                    <div
                      key={agent.name}
                      className={`rounded-2xl overflow-hidden cursor-pointer transition-all ${
                        selectedAgent === agent.name 
                          ? "ring-4 ring-violet-600 shadow-lg" 
                          : "hover:shadow-md"
                      }`}
                      onClick={() => handleAgentClick(agent.name)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          handleAgentClick(agent.name);
                        }
                      }}
                      tabIndex={0}
                      role="button"
                      aria-pressed={selectedAgent === agent.name}
                    >
                      <div className="relative h-32 sm:h-40 bg-gray-100">
                        <Image
                          src={agent.imageSrc}
                          alt={agent.name}
                          layout="fill"
                          objectFit="cover"
                        />
                      </div>
                      <div className="p-3">
                        <h3 className="text-neutral-800 text-sm sm:text-base font-medium font-['Geist'] leading-normal">{agent.name}</h3>
                        <p className="text-neutral-500 text-xs sm:text-sm font-normal font-['Geist'] leading-[14px]">Edited {agent.lastEdited}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
          
          <div className="p-4 sm:p-6 border-t border-slate-200 flex justify-between gap-4">
            <Button
              onClick={onCancel}
              variant="outline"
              size="default"
              className="w-full sm:flex-1"
            >
              Back
            </Button>
            <Button
              onClick={onNext}
              disabled={!selectedAgent}
              variant="default"
              size="default"
              className="w-full sm:flex-1 text-white bg-neutral-800 hover:bg-neutral-900"
            >
              Next
            </Button>
          </div>
        </>
      )}
    </div>
  );
};
