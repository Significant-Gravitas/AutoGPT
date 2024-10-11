import * as React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

interface CreatorCardProps {
  creatorName: string;
  bio: string;
  agentsUploaded: number;
  onClick: () => void;
  avatarSrc: string;
}

export const CreatorCard: React.FC<CreatorCardProps> = ({
  creatorName,
  bio,
  agentsUploaded,
  onClick,
  avatarSrc,
}) => {
  const [isHovered, setIsHovered] = React.useState(false);

  const handleMouseEnter = () => setIsHovered(true);
  const handleMouseLeave = () => setIsHovered(false);

  return (
    <div
      className={`relative h-96 w-[210px] ${isHovered ? "shadow-lg" : ""} rounded-xl transition-shadow duration-300`}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="absolute left-0 top-0 h-[238px] w-[210px] rounded-xl bg-[#d9d9d9]" />
      <Avatar className="absolute left-[16px] top-[158px] h-16 w-16">
        <AvatarImage src={avatarSrc} alt={creatorName} />
        <AvatarFallback>{creatorName.charAt(0)}</AvatarFallback>
      </Avatar>
      <div className="font-neue absolute left-0 top-[254px] text-xl font-bold tracking-tight text-[#272727]">
        {creatorName}
      </div>
      <div className="font-neue absolute left-0 top-[284px] line-clamp-3 w-[210px] text-base font-normal leading-[21px] tracking-tight text-[#282828]">
        {bio}
      </div>
      <div className="font-neue absolute left-0 top-[360px] text-base font-medium tracking-tight text-[#272727]">
        {agentsUploaded} agents uploaded
      </div>
    </div>
  );
};
