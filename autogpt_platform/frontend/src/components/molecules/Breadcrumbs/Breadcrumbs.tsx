import * as React from "react";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";

interface BreadcrumbItem {
  name: string;
  link: string;
}

interface Props {
  items: BreadcrumbItem[];
}

export function Breadcrumbs({ items }: Props) {
  return (
    <div className="flex items-center gap-4">
      <div className="flex h-auto flex-wrap items-center justify-start gap-3 rounded-[5rem] dark:bg-transparent">
        {items.map((item, index) => (
          <React.Fragment key={index}>
            <Link
              href={item.link}
              className="text-zinc-700 transition-colors hover:text-zinc-900 hover:no-underline"
            >
              {item.name}
            </Link>
            {index < items.length - 1 && (
              <Text variant="body-medium" className="text-zinc-700">
                /
              </Text>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
