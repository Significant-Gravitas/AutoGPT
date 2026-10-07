import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import * as React from "react";

interface BreadcrumbItem {
  name: string;
  link?: string;
}

interface Props {
  items: BreadcrumbItem[];
}

export function Breadcrumbs({ items }: Props) {
  return (
    <div className="mb-4 flex h-auto flex-wrap items-center justify-start gap-2 md:mb-0 md:gap-2">
      {items.map((item, index) => (
        <React.Fragment key={index}>
          {item.link ? (
            <Link
              href={item.link}
              className="text-sm font-normal text-muted-foreground transition-colors hover:text-foreground hover:no-underline"
            >
              {item.name}
            </Link>
          ) : (
            <span className="text-sm font-normal text-foreground">
              {item.name}
            </span>
          )}
          {index < items.length - 1 && (
            <Text variant="small-medium" className="text-muted-foreground">
              /
            </Text>
          )}
        </React.Fragment>
      ))}
    </div>
  );
}
