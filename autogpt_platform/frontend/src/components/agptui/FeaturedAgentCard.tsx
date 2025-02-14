import Image from "next/image";
import { StarRatingIcons } from "@/components/ui/icons";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useState } from "react";
import { StoreAgent } from "@/lib/autogpt-server-api";

interface FeaturedStoreCardProps {
  agent: StoreAgent;
  backgroundColor: string;
}

export const FeaturedAgentCard: React.FC<FeaturedStoreCardProps> = ({
  agent,
  backgroundColor,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <Card
      data-testid="featured-store-card"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      className={`h-[482px] w-[440px] flex flex-col ${backgroundColor}`}
    >
      <CardHeader className="flex-none space-y-3 pb-2">
        <CardTitle className="text-xl leading-tight">{agent.agent_name}</CardTitle>
        <CardDescription className="text-sm">By {agent.creator}</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 relative p-4">
        <div className="absolute inset-0 m-4 overflow-hidden rounded-xl">
          <div className={`h-full w-full transition-opacity duration-200 ${isHovered ? "opacity-0" : "opacity-100"}`}>
            <Image
              src={agent.agent_image || "/AUTOgpt_Logo_dark.png"}
              alt={`${agent.agent_name} preview`}
              fill
              sizes="100%"
              className="rounded-xl object-cover"
            />
          </div>
          <div 
            className={`absolute inset-0 p-4 overflow-y-auto transition-opacity duration-200 ${
              isHovered ? "opacity-100" : "opacity-0"
            }`}
          >
            <CardDescription className="text-sm leading-relaxed text-muted-foreground line-clamp-[11]">
              {agent.description}
            </CardDescription>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex-none h-[60px] flex items-center justify-between">
        <div className="font-semibold">
          {agent.runs?.toLocaleString() ?? "0"} runs
        </div>
        <div className="flex items-center gap-1.5">
          <p>{agent.rating.toFixed(1) ?? "0.0"}</p>
          <div
            className="inline-flex items-center justify-start gap-px"
            role="img"
            aria-label={`Rating: ${agent.rating.toFixed(1)} out of 5 stars`}
          >
            {StarRatingIcons(agent.rating)}
          </div>
        </div>
      </CardFooter>
    </Card>
  );
};
