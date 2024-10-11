import * as React from "react";
import { StarIcon, StarFilledIcon } from "@radix-ui/react-icons";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

interface StoreCardProps {
  agentName: string;
  description: string;
  runs: number;
  rating: number;
  onClick: () => void;
  avatarSrc: string;
}

export const StoreCard: React.FC<StoreCardProps> = ({
  agentName,
  description,
  runs,
  rating,
  onClick,
  avatarSrc,
}) => {
  const [isHovered, setIsHovered] = React.useState(false);

  const handleMouseEnter = () => setIsHovered(true);
  const handleMouseLeave = () => setIsHovered(false);

  const renderStars = () => {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 !== 0;
    const stars = [];

    for (let i = 0; i < 5; i++) {
      if (i < fullStars) {
        stars.push(<StarFilledIcon key={i} className="text-black" />);
      } else if (i === fullStars && hasHalfStar) {
        stars.push(<StarIcon key={i} className="text-black" />);
      } else {
        stars.push(<StarIcon key={i} className="text-black" />);
      }
    }

    return stars;
  };

  return (
    <div
      className={`w-110 relative h-96 pb-2 ${isHovered ? "shadow-lg" : ""} rounded-xl transition-shadow duration-300`}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="h-69 w-110 absolute left-0 top-0 rounded-xl bg-[#d9d9d9]" />
      <Avatar className="top-39 absolute left-4 h-16 w-16">
        <AvatarImage src={avatarSrc} alt={agentName} />
        <AvatarFallback>{agentName.charAt(0)}</AvatarFallback>
      </Avatar>
      <div className="font-neue top-63 absolute pl-2 text-xl font-bold tracking-tight text-[#272727]">
        {agentName}
      </div>
      <div className="font-neue top-68 w-110 absolute pl-2 text-base font-normal leading-[21px] tracking-tight text-[#282828]">
        {description}
      </div>
      <div className="font-neue top-89 absolute pl-2 text-base font-medium tracking-tight text-[#272727]">
        {runs.toLocaleString()}+ runs
      </div>
      <div className="left-77 top-89 absolute pb-2">
        <div className="font-neue absolute top-0 pl-2 text-base font-medium tracking-tight text-[#272727]">
          {rating.toFixed(1)}
        </div>
        <div className="left-8.5 h-7.5 absolute top-[2px] inline-flex items-center justify-start gap-px">
          {renderStars()}
        </div>
      </div>
    </div>
  );
};
