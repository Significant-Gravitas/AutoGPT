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
          // TODO(tokens): no exact text token for tracking-[0.14em]
          className="uppercase tracking-[0.14em] !text-foreground-tertiary"
        >
          {itemsTitle}
        </Text>
      ) : null}
      <ul className="flex max-w-lg flex-col gap-3">
        {items.map((item) => (
          <FeatureListItem key={item.title} item={item} />
        ))}
      </ul>
    </div>
  );
}
