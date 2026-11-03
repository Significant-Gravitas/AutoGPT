import { BlockInfoCategoriesItem } from "@/app/api/__generated__/models/blockInfoCategoriesItem";
import { Badge } from "@/components/atoms/Badge/Badge";
import { beautifyString, cn } from "@/lib/utils";

export const NodeBadges = ({
  categories,
}: {
  categories: BlockInfoCategoriesItem[];
}) => {
  return categories.map((category) => (
    <Badge
      key={category.category}
      variant="solid"
      className={cn(
        "rounded-full border border-foreground-tertiary bg-surface-hover text-foreground-base shadow-none",
      )}
    >
      {beautifyString(category.category.toLowerCase())}
    </Badge>
  ));
};
