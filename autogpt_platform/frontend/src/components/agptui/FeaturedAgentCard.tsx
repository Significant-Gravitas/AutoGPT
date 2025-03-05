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
      className={`flex h-full flex-col ${backgroundColor}`}
    >
      <CardHeader>
        <CardTitle className="line-clamp-2 text-base sm:text-xl">
          {agent.agent_name}
        </CardTitle>
        <CardDescription className="text-sm">
          By {agent.creator}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex-1 p-4">
        <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl">
          <Image
            src={agent.agent_image || "/AUTOgpt_Logo_dark.png"}
            alt={`${agent.agent_name} preview`}
            fill
            sizes="100%"
            className={`object-cover transition-opacity duration-200 ${
              isHovered ? "opacity-0" : "opacity-100"
            }`}
          />
          <div
            className={`absolute inset-0 overflow-y-auto p-4 transition-opacity duration-200 ${
              isHovered ? "opacity-100" : "opacity-0"
            }`}
          >
            <CardDescription className="line-clamp-[6] text-xs sm:line-clamp-[8] sm:text-sm">
              {agent.description}
            </CardDescription>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex items-center justify-between">
        <div className="font-semibold">
          {agent.runs?.toLocaleString() ?? "0"} runs
        </div>
        <div className="flex items-center gap-1.5">
          <p>{agent.rating.toFixed(1) ?? "0.0"}</p>
          {StarRatingIcons(agent.rating)}
        </div>
      </CardFooter>
    </Card>
  );
};
