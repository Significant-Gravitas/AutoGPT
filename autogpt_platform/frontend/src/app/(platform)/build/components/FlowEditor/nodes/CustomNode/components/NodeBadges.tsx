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
        // TODO(tokens): no exact role: border-slate-500 #64748B, bg-slate-100 #EFF1F4
        "rounded-full border border-slate-500 bg-slate-100 text-foreground-base shadow-none",
      )}
    >
      {beautifyString(category.category.toLowerCase())}
    </Badge>
  ));
};
