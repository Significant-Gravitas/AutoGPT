"use client";

import React from "react";
import { Star, Download, ArrowRight, Info } from "lucide-react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";

interface Agent {
  id: string;
  version: string;
  name: string;
  description: string;
  creator?: string;
  rating?: number;
  runs?: number;
  downloads?: number;
  categories?: string[];
}

interface AgentDiscoveryCardProps {
  agents: Agent[];
  onSelectAgent: (agent: Agent) => void;
  onGetDetails: (agent: Agent) => void;
  className?: string;
}

export function AgentDiscoveryCard({
  agents,
  onSelectAgent,
  onGetDetails,
  className,
}: AgentDiscoveryCardProps) {
  if (!agents || agents.length === 0) {
    return null;
  }

  return (
    <div className={cn("my-4 space-y-3", className)}>
      <div className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
        🎯 Recommended Agents for You:
      </div>
      
      <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        {agents.slice(0, 3).map((agent) => (
          <div
            key={`${agent.id}-${agent.version}`}
            className={cn(
              "group relative overflow-hidden rounded-lg border",
              "border-neutral-200 dark:border-neutral-700",
              "bg-white dark:bg-neutral-900",
              "transition-all duration-300 hover:shadow-lg",
              "animate-in fade-in-50 slide-in-from-bottom-2"
            )}
          >
            <div className="bg-gradient-to-br from-violet-500/10 to-purple-500/10 p-4">
              <div className="mb-2 flex items-start justify-between">
                <h3 className="font-semibold text-neutral-900 dark:text-neutral-100">
                  {agent.name}
                </h3>
                {agent.rating && (
                  <div className="flex items-center gap-1 text-sm">
                    <Star className="h-3.5 w-3.5 fill-yellow-400 text-yellow-400" />
                    <span className="text-neutral-600 dark:text-neutral-400">
                      {agent.rating.toFixed(1)}
                    </span>
                  </div>
                )}
              </div>
              
              <p className="mb-3 line-clamp-2 text-sm text-neutral-600 dark:text-neutral-400">
                {agent.description}
              </p>
              
              {agent.creator && (
                <p className="mb-2 text-xs text-neutral-500 dark:text-neutral-500">
                  by {agent.creator}
                </p>
              )}
              
              <div className="mb-3 flex items-center gap-3 text-xs text-neutral-500 dark:text-neutral-500">
                {agent.runs && (
                  <div className="flex items-center gap-1">
                    <Download className="h-3 w-3" />
                    {agent.runs.toLocaleString()} runs
                  </div>
                )}
                {agent.downloads && (
                  <div className="flex items-center gap-1">
                    <Download className="h-3 w-3" />
                    {agent.downloads.toLocaleString()} downloads
                  </div>
                )}
              </div>
              
              {agent.categories && agent.categories.length > 0 && (
                <div className="mb-3 flex flex-wrap gap-1">
                  {agent.categories.slice(0, 3).map((category) => (
                    <span
                      key={category}
                      className="rounded-full bg-neutral-100 dark:bg-neutral-800 px-2 py-0.5 text-xs text-neutral-600 dark:text-neutral-400"
                    >
                      {category}
                    </span>
                  ))}
                </div>
              )}
              
              <div className="flex gap-2">
                <Button
                  onClick={() => onGetDetails(agent)}
                  variant="secondary"
                  size="sm"
                  className="flex-1"
                >
                  <Info className="mr-1 h-3 w-3" />
                  Details
                </Button>
                <Button
                  onClick={() => onSelectAgent(agent)}
                  variant="primary"
                  size="sm"
                  className="flex-1"
                >
                  Set Up
                  <ArrowRight className="ml-1 h-3 w-3" />
                </Button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}