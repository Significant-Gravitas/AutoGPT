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
      className={`flex h-[30rem] w-full min-w-[94vw] max-w-[27.5rem] flex-col hover:cursor-pointer md:w-[24rem] md:min-w-0 lg:w-[27.5rem] ${backgroundColor} rounded-[1.5rem] border-none px-5 pb-5 pt-6 transition-colors duration-200`}
    >
      <CardHeader className="mb-7 h-[9.5rem] space-y-3 p-0">
        <CardTitle className="line-clamp-3 font-poppins text-3xl font-medium text-zinc-800">
          {agent.agent_name}
        </CardTitle>
        <CardDescription className="line-clamp-1 font-sans text-base font-normal text-zinc-800">
          By {agent.creator}
        </CardDescription>
      </CardHeader>

      <CardContent className="mb-4 flex flex-1 flex-col gap-4 p-0">
        <div className="relative flex-1 overflow-hidden rounded-xl">
          <Image
            src={agent.agent_image || "/default_agent_image.jpg"}
            alt={`${agent.agent_name} preview`}
            fill
            sizes="100%"
            className={`object-cover transition-opacity duration-200 ${
              isHovered ? "opacity-0" : "opacity-100"
            }`}
          />
          <Image
            src={agent.creator_avatar || "/default_avatar.png"}
            alt={`${agent.creator} avatar`}
            width={50}
            height={50}
            className={`absolute bottom-3 left-3 aspect-square rounded-full border border-zinc-200 transition-opacity duration-200 ${
              isHovered ? "opacity-0" : "opacity-100"
            }`}
          />
          <div
            className={`absolute inset-0 overflow-hidden p-0 transition-opacity duration-200 ${
              isHovered ? "opacity-100" : "opacity-0"
            }`}
          >
            <CardDescription
              data-testid="agent-description"
              className="line-clamp-6 font-sans text-sm text-zinc-600"
            >
              {agent.description}
            </CardDescription>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex min-h-7 flex-col items-start justify-between p-0 sm:flex-row sm:items-center">
        <div className="font-sans text-base font-medium text-zinc-800">
          {agent.runs?.toLocaleString() ?? "0"} runs
        </div>
        <div className="flex items-center gap-1.5 font-sans text-base font-medium text-zinc-800">
          <p>{agent.rating.toFixed(1) ?? "0.0"}</p>
          {StarRatingIcons(agent.rating)}
        </div>
      </CardFooter>
    </Card>
  );
};
