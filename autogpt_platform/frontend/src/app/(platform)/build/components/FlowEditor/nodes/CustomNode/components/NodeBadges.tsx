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
      variant="info"
      className={cn(
        "rounded-full border border-slate-500 bg-slate-100 px-2.5 font-semibold text-black ring-0",
      )}
    >
      {beautifyString(category.category.toLowerCase())}
    </Badge>
  ));
};
