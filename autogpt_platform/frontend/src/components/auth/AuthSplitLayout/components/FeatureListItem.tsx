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
    <li className="flex items-start gap-4 rounded-xl border border-white/5 bg-white/[0.03] p-4 backdrop-blur-sm">
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500/30 to-purple-500/30 text-white ring-1 ring-white/10">
        {item.icon}
      </span>
      <div className="flex flex-col gap-0.5">
        <Text variant="body-medium" className="!text-white">
          {item.title}
        </Text>
        {item.description ? (
          <Text variant="small" className="!text-slate-400">
            {item.description}
          </Text>
        ) : null}
      </div>
    </li>
  );
}
