import * as React from "react";
import Link from "next/link";
import { IconLeftArrow, IconRightArrow } from "@/components/ui/icons";

interface BreadcrumbItem {
  name: string;
  link: string;
}

interface BreadCrumbsProps {
  items: BreadcrumbItem[];
}

export const BreadCrumbs: React.FC<BreadCrumbsProps> = ({ items }) => {
  return (
    <div className="flex h-auto flex-wrap items-center justify-start gap-2.5 bg-transparent">
      {items.map((item, index) => (
        <React.Fragment key={index}>
          <Link href={item.link}>
            <span className="font-sans text-base font-medium text-zinc-800 transition-colors duration-200 hover:text-zinc-400 dark:text-neutral-100 dark:hover:text-gray-500">
              {item.name.length > 50
                ? `${item.name.slice(0, 50)}...`
                : item.name}
            </span>
          </Link>
          {index < items.length - 1 && (
            <span className="font-sans text-base font-medium text-zinc-800 dark:text-zinc-100">
              /
            </span>
          )}
        </React.Fragment>
      ))}
    </div>
  );
};
