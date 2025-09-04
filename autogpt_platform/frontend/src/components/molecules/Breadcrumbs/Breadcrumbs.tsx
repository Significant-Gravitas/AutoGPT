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
    <div className="mb-4 flex h-auto flex-wrap items-center justify-start gap-2 md:mb-0 md:gap-3">
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
  );
}
