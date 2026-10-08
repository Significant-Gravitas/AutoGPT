import { Link } from "@/components/atoms/Link/Link";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Fragment } from "react";

interface BreadcrumbItem {
  name: string;
  link?: string;
}

interface Props {
  items: BreadcrumbItem[];
}

export function Breadcrumbs({ items }: Props) {
  return (
    <Breadcrumb className="mb-4 md:mb-0">
      <BreadcrumbList className="gap-2">
        {items.map((item, index) => (
          <Fragment key={index}>
            <BreadcrumbItem>
              {item.link ? (
                <BreadcrumbLink
                  className="font-normal text-muted-foreground hover:no-underline"
                  render={<Link href={item.link}>{item.name}</Link>}
                />
              ) : (
                <BreadcrumbPage>{item.name}</BreadcrumbPage>
              )}
            </BreadcrumbItem>
            {index < items.length - 1 && <BreadcrumbSeparator />}
          </Fragment>
        ))}
      </BreadcrumbList>
    </Breadcrumb>
  );
}
