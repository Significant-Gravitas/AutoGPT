import { Text } from "@/components/atoms/Text/Text";
import { FeatureItem, FeatureListItem } from "./FeatureListItem";

interface Props {
  itemsTitle?: string;
  items: FeatureItem[];
}

export function FeatureList({ itemsTitle, items }: Props) {
  return (
    <div className="flex flex-col gap-2">
      {itemsTitle ? (
        <Text
          variant="small-medium"
          className="tracking-[0.14em] text-slate-400 uppercase"
        >
          {itemsTitle}
        </Text>
      ) : null}
      <ul className="flex flex-col gap-3">
        {items.map((item) => (
          <FeatureListItem key={item.title} item={item} />
        ))}
      </ul>
    </div>
  );
}
