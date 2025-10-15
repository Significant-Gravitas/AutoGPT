"use client";

import React, { useState, useRef, useEffect } from "react";
import Image from "next/image";
import {
  ChevronLeft,
  ChevronRight,
  Star,
  Play,
  Info,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";

interface Agent {
  id: string;
  name: string;
  sub_heading: string;
  description: string;
  creator: string;
  creator_avatar?: string;
  agent_image?: string;
  rating?: number;
  runs?: number;
}

interface AgentCarouselProps {
  agents: Agent[];
  query: string;
  onSelectAgent: (agent: Agent) => void;
  onGetDetails: (agent: Agent) => void;
  className?: string;
}

export function AgentCarousel({
  agents,
  query,
  onSelectAgent,
  onGetDetails,
  className,
}: AgentCarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAutoScrolling, setIsAutoScrolling] = useState(true);
  const carouselRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // Deduplicate agents by ID
  const uniqueAgents = React.useMemo(() => {
    const seen = new Set<string>();
    return agents.filter((agent) => {
      if (seen.has(agent.id)) {
        return false;
      }
      seen.add(agent.id);
      return true;
    });
  }, [agents]);

  // Auto-scroll effect
  useEffect(() => {
    if (!isAutoScrolling || uniqueAgents.length <= 3) return;

    const timer = setInterval(() => {
      setCurrentIndex(
        (prev) => (prev + 1) % Math.max(1, uniqueAgents.length - 2),
      );
    }, 5000);

    return () => clearInterval(timer);
  }, [isAutoScrolling, uniqueAgents.length]);

  // Scroll to current index
  useEffect(() => {
    if (scrollContainerRef.current) {
      const cardWidth = 320; // Approximate card width including gap
      scrollContainerRef.current.scrollTo({
        left: currentIndex * cardWidth,
        behavior: "smooth",
      });
    }
  }, [currentIndex]);

  const handlePrevious = () => {
    setIsAutoScrolling(false);
    setCurrentIndex((prev) => Math.max(0, prev - 1));
  };

  const handleNext = () => {
    setIsAutoScrolling(false);
    setCurrentIndex((prev) =>
      Math.min(Math.max(0, uniqueAgents.length - 3), prev + 1),
    );
  };

  const handleDotClick = (index: number) => {
    setIsAutoScrolling(false);
    setCurrentIndex(index);
  };

  if (!uniqueAgents || uniqueAgents.length === 0) {
    return null;
  }

  const maxVisibleIndex = Math.max(0, uniqueAgents.length - 3);

  return (
    <div className={cn("my-6 space-y-4", className)} ref={carouselRef}>
      {/* Header */}
      <div className="flex items-center justify-between px-4">
        <div className="flex items-center gap-2">
          <Sparkles className="h-5 w-5 text-violet-600" />
          <h3 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">
            Found {uniqueAgents.length} agents for &ldquo;{query}&rdquo;
          </h3>
        </div>
        {uniqueAgents.length > 3 && (
          <div className="flex items-center gap-2">
            <Button
              onClick={handlePrevious}
              variant="secondary"
              size="small"
              disabled={currentIndex === 0}
              className="p-1"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              onClick={handleNext}
              variant="secondary"
              size="small"
              disabled={currentIndex >= maxVisibleIndex}
              className="p-1"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}
      </div>

      {/* Carousel Container */}
      <div className="relative overflow-hidden px-4">
        <div
          ref={scrollContainerRef}
          className="scrollbar-hide flex gap-4 overflow-x-auto scroll-smooth"
          style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
        >
          {uniqueAgents.map((agent) => (
            <div
              key={agent.id}
              className={cn(
                "w-[300px] flex-shrink-0",
                "group relative overflow-hidden rounded-xl",
                "border border-neutral-200 dark:border-neutral-700",
                "bg-white dark:bg-neutral-900",
                "transition-all duration-300",
                "hover:scale-[1.02] hover:shadow-xl",
                "animate-in fade-in-50 slide-in-from-bottom-2",
              )}
            >
              {/* Agent Image Header */}
              <div className="relative h-32 bg-gradient-to-br from-violet-500/20 via-purple-500/20 to-indigo-500/20">
                {agent.agent_image ? (
                  <Image
                    src={agent.agent_image}
                    alt={agent.name}
                    fill
                    className="object-cover opacity-90"
                  />
                ) : (
                  <div className="flex h-full items-center justify-center">
                    <div className="text-4xl">🤖</div>
                  </div>
                )}
                {agent.rating && (
                  <div className="absolute right-2 top-2 flex items-center gap-1 rounded-full bg-black/50 px-2 py-1 backdrop-blur">
                    <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                    <span className="text-xs font-medium text-white">
                      {agent.rating.toFixed(1)}
                    </span>
                  </div>
                )}
              </div>

              {/* Agent Content */}
              <div className="space-y-3 p-4">
                <div>
                  <h4 className="line-clamp-1 font-semibold text-neutral-900 dark:text-neutral-100">
                    {agent.name}
                  </h4>
                  {agent.sub_heading && (
                    <p className="mt-0.5 line-clamp-1 text-xs text-violet-600 dark:text-violet-400">
                      {agent.sub_heading}
                    </p>
                  )}
                </div>

                <p className="line-clamp-2 text-sm text-neutral-600 dark:text-neutral-400">
                  {agent.description}
                </p>

                {/* Creator Info */}
                <div className="flex items-center gap-2 text-xs text-neutral-500">
                  {agent.creator_avatar ? (
                    <div className="relative h-4 w-4">
                      <Image
                        src={agent.creator_avatar}
                        alt={agent.creator}
                        fill
                        className="rounded-full object-cover"
                      />
                    </div>
                  ) : (
                    <div className="h-4 w-4 rounded-full bg-neutral-300 dark:bg-neutral-600" />
                  )}
                  <span>by {agent.creator}</span>
                  {agent.runs && (
                    <>
                      <span className="text-neutral-400">•</span>
                      <span>{agent.runs.toLocaleString()} runs</span>
                    </>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="flex gap-2 pt-2">
                  <Button
                    onClick={() => onGetDetails(agent)}
                    variant="secondary"
                    size="small"
                    className="flex-1"
                  >
                    <Info className="mr-1 h-3 w-3" />
                    Details
                  </Button>
                  <Button
                    onClick={() => onSelectAgent(agent)}
                    variant="primary"
                    size="small"
                    className="flex-1"
                  >
                    <Play className="mr-1 h-3 w-3" />
                    Set Up
                  </Button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Pagination Dots */}
      {agents.length > 3 && (
        <div className="flex justify-center gap-1.5 px-4">
          {Array.from({ length: maxVisibleIndex + 1 }).map((_, index) => (
            <button
              key={index}
              onClick={() => handleDotClick(index)}
              className={cn(
                "h-1.5 rounded-full transition-all duration-300",
                index === currentIndex
                  ? "w-6 bg-violet-600"
                  : "w-1.5 bg-neutral-300 hover:bg-neutral-400 dark:bg-neutral-600",
              )}
              aria-label={`Go to slide ${index + 1}`}
            />
          ))}
        </div>
      )}
    </div>
  );
}
