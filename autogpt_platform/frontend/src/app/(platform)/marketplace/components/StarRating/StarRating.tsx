import { Icon } from "@/components/atoms/Icon/Icon";
import { StarIcon } from "@hugeicons/core-free-icons";

const STAR_POSITIONS = [1, 2, 3, 4, 5];

interface Props {
  rating: number;
}

export function StarRating({ rating }: Props) {
  const clampedRating = Math.max(0, Math.min(5, rating));

  return (
    <>
      {STAR_POSITIONS.map((position) => (
        <Icon
          key={position}
          icon={StarIcon}
          size={16}
          fill={position <= clampedRating ? "currentColor" : "none"}
          className="text-black"
          aria-hidden
        />
      ))}
    </>
  );
}
