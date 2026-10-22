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
    <li className="flex items-start gap-4 rounded-2xl bg-white p-4 smooth-shadow-ring-sm">
      <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-violet-50 text-violet-600">
        {item.icon}
      </span>
      <div className="flex flex-col gap-0.5">
        <Text variant="body-medium" className="!text-zinc-900">
          {item.title}
        </Text>
        {item.description ? (
          <Text variant="small" className="!text-zinc-500">
            {item.description}
          </Text>
        ) : null}
      </div>
    </li>
  );
}
