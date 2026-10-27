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
    <div
      data-slot="breadcrumbs"
      className="mb-4 flex h-auto flex-wrap items-center justify-start gap-2 md:mb-0 md:gap-2"
    >
      {items.map((item, index) => (
        <React.Fragment key={index}>
          {item.link ? (
            <Link
              href={item.link}
              data-slot="breadcrumbs-link"
              className="text-body font-normal leading-5 text-foreground-secondary transition-colors hover:text-foreground-strong hover:no-underline"
            >
              {item.name}
            </Link>
          ) : (
            <span
              data-slot="breadcrumbs-current"
              className="text-body font-normal leading-5 text-foreground-strong"
            >
              {item.name}
            </span>
          )}
          {index < items.length - 1 && (
            <Text
              variant="small-medium"
              data-slot="breadcrumbs-separator"
              className="text-foreground-secondary"
            >
              /
            </Text>
          )}
        </React.Fragment>
      ))}
    </div>
  );
}
