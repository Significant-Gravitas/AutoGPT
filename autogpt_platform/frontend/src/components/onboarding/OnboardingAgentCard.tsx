import { cn } from "@/lib/utils";
import Image from "next/image";
import { FaRegStar, FaStar, FaStarHalfAlt } from "react-icons/fa";

interface OnboardingAgentCardProps {
  id: string;
  image: string;
  name: string;
  description: string;
  author: string;
  runs: number;
  rating: number;
  selected?: boolean;
  onClick: () => void;
}

function StarRating({ rating }: { rating: number }) {
  // Round to 1 decimal place
  const roundedRating = Math.round(rating * 10) / 10;

  // Generate array of 5 star values
  const stars = Array(5)
    .fill(0)
    .map((_, index) => {
      const starValue = index + 1;
      const difference = roundedRating - index;

      if (difference >= 1) {
        return "full";
      } else if (difference > 0) {
        // Half star for values between 0.2 and 0.8
        return difference >= 0.8
          ? "full"
          : difference >= 0.2
            ? "half"
            : "empty";
      }
      return "empty";
    });

  return (
    <div className="font-geist flex items-center gap-0.5 text-sm font-medium text-zinc-800">
      {/* Display numerical rating */}
      <span className="mr-1 mt-1">{roundedRating}</span>

      {/* Display stars */}
      {stars.map((starType, index) => {
        if (starType === "full") {
          return <FaStar size={15} key={index} />;
        } else if (starType === "half") {
          return <FaStarHalfAlt size={15} key={index} />;
        } else {
          return <FaRegStar size={15} key={index} />;
        }
      })}
    </div>
  );
}

export default function OnboardingAgentCard({
  id,
  image,
  name,
  description,
  author,
  runs,
  rating,
  selected,
  onClick,
}: OnboardingAgentCardProps) {
  return (
    <div
      className={cn(
        "relative cursor-pointer transition-all duration-200 ease-in-out",
        "h-[394px] w-[368px] rounded-xl border border-transparent bg-white",
        selected ? "bg-[#F5F3FF80]" : "hover:border-zinc-400",
      )}
      onClick={onClick}
    >
      {/* Image container with relative positioning for profile pic overlay */}
      <div className="relative">
        <Image
          src={image}
          alt="Agent cover"
          className="m-2 h-[196px] w-[350px] rounded-xl object-cover"
          width={350}
          height={196}
        />
        {/* Profile picture overlay */}
        <div className="absolute bottom-2 left-4">
          <Image
            src={image}
            alt="Profile picture"
            className="h-[50px] w-[50px] rounded-full border border-white object-cover object-center"
            width={50}
            height={50}
          />
        </div>
      </div>

      {/* Content container */}
      <div className="flex h-[180px] flex-col justify-between px-4 pb-3">
        {/* Text content wrapper */}
        <div>
          {/* Title - allows 2 lines max */}
          <p className="text-md font-geist line-clamp-2 max-h-[50px] text-base font-medium leading-normal text-zinc-800">
            {name}
          </p>

          {/* Author - single line with truncate */}
          <p className="truncate text-sm font-normal leading-normal text-zinc-600">
            by {author}
          </p>

          {/* Description - flexible with ellipsis */}
          <p
            className={cn(
              "mt-2 line-clamp-3 text-sm leading-5",
              selected ? "text-zinc-500" : "text-zinc-400",
            )}
          >
            {description}
          </p>
        </div>

        {/* Bottom stats */}
        <div className="flex w-full items-center justify-between">
          <span className="font-geist mt-1 text-sm font-medium text-zinc-800">
            {runs.toLocaleString("en-US")} runs
          </span>
          <StarRating rating={rating} />
        </div>
      </div>
      <div
        className={cn(
          "pointer-events-none absolute inset-0 rounded-xl border-2 transition-all duration-200 ease-in-out",
          selected ? "border-violet-700" : "border-transparent",
        )}
      />
    </div>
  );
}
