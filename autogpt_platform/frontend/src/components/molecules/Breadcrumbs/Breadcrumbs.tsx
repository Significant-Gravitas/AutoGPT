import * as React from "react";
import Link from "next/link";

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
      <div className="flex h-auto flex-wrap items-center justify-start gap-4 rounded-[5rem] dark:bg-transparent">
        {items.map((item, index) => (
          <React.Fragment key={index}>
            <Link href={item.link}>
              <span className="rounded py-1 pr-2 text-xl font-medium leading-9 tracking-tight text-[#272727] transition-colors duration-200 hover:text-gray-400 dark:text-neutral-100 dark:hover:text-gray-500">
                {item.name}
              </span>
            </Link>
            {index < items.length - 1 && (
              <span className="text-center text-2xl font-normal text-black dark:text-neutral-100">
                /
              </span>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
