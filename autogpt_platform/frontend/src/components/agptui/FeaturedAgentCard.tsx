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
      className={`flex h-[47rem] w-full min-w-96 max-w-[27.5rem] flex-col md:w-[24rem] lg:w-[27.5rem] ${backgroundColor} rounded-[1.5rem] border-none px-5 pb-5 pt-7 transition-colors duration-200`}
    >
      <CardHeader className="mb-7 min-h-48 space-y-3 p-0">
        <CardTitle className="line-clamp-3 font-poppins text-4xl font-medium text-neutral-900">
          {agent.agent_name}
        </CardTitle>
        <CardDescription className="line-clamp-2 font-sans text-xl font-normal text-neutral-800">
          {agent.sub_heading}
        </CardDescription>
      </CardHeader>

      <CardContent className="mb-4 flex flex-1 flex-col gap-4 p-0">
        <p className="line-clamp-1 font-sans text-xl text-neutral-800">
          By {agent.creator}
        </p>
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
            width={74}
            height={74}
            className={`absolute bottom-3 left-3 aspect-square rounded-full transition-opacity duration-200 ${
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
              className="line-clamp-6 font-sans text-base text-neutral-800"
            >
              {agent.description}
            </CardDescription>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex min-h-7 items-center justify-between p-0">
        <div className="font-sans text-lg font-semibold text-neutral-800">
          {agent.runs?.toLocaleString() ?? "0"} runs
        </div>
        <div className="flex items-center gap-1.5 font-sans text-lg font-semibold text-neutral-800">
          <p>{agent.rating.toFixed(1) ?? "0.0"}</p>
          {StarRatingIcons(agent.rating)}
        </div>
      </CardFooter>
    </Card>
  );
};
