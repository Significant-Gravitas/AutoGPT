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
      className={`relative h-96 w-[440px] ${isHovered ? "shadow-lg" : ""} rounded-xl transition-shadow duration-300`}
      onClick={onClick}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <div className="absolute left-0 top-0 h-[238px] w-[440px] rounded-xl bg-[#d9d9d9]" />
      <Avatar className="absolute left-[16px] top-[158px] h-16 w-16">
        <AvatarImage src={avatarSrc} alt={agentName} />
        <AvatarFallback>{agentName.charAt(0)}</AvatarFallback>
      </Avatar>
      <div className="font-neue absolute left-0 top-[254px] text-xl font-bold tracking-tight text-[#272727]">
        {agentName}
      </div>
      <div className="font-neue absolute left-0 top-[284px] w-[440px] text-base font-normal leading-[21px] tracking-tight text-[#282828]">
        {description}
      </div>
      <div className="font-neue absolute left-0 top-[360px] text-base font-medium tracking-tight text-[#272727]">
        {runs.toLocaleString()}+ runs
      </div>
      <div className="absolute left-[297px] top-[360px] pb-2">
        <div className="font-neue absolute left-0 top-0 text-base font-medium tracking-tight text-[#272727]">
          {rating.toFixed(1)}
        </div>
        <div className="absolute left-[34px] top-0 inline-flex h-[19px] items-center justify-start gap-px">
          {renderStars()}
        </div>
      </div>
    </div>
  );
};
