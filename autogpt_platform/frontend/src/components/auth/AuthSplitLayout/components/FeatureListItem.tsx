import { ReactNode } from "react";
import { Text } from "@/components/atoms/Text/Text";

export interface FeatureItem {
  icon: ReactNode;
  title: string;
  description?: string;
}

interface Props {
  item: FeatureItem;
}

export function FeatureListItem({ item }: Props) {
  return (
    <li className="flex items-start gap-4 rounded-2xl bg-surface p-4 smooth-shadow-ring-sm">
      <span
        // TODO(tokens): no exact role for text-violet-600 (#7C3AED)
        className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-accent-surface text-violet-600"
      >
        {item.icon}
      </span>
      <div className="flex flex-col gap-0.5">
        <Text variant="body-medium" className="!text-foreground-strong">
          {item.title}
        </Text>
        {item.description ? (
          <Text variant="small" className="!text-foreground-tertiary">
            {item.description}
          </Text>
        ) : null}
      </div>
    </li>
  );
}
