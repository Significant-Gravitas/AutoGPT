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
    <div className="flex items-center gap-4">
      {/*
      Commented out for now, but keeping until we have approval to remove
      <button className="flex h-12 w-12 items-center justify-center rounded-full border border-neutral-200 transition-colors hover:bg-neutral-50 dark:border-neutral-700 dark:hover:bg-neutral-800">
        <IconLeftArrow className="h-5 w-5 text-neutral-900 dark:text-neutral-100" />
      </button>
      <button className="flex h-12 w-12 items-center justify-center rounded-full border border-neutral-200 transition-colors hover:bg-neutral-50 dark:border-neutral-700 dark:hover:bg-neutral-800">
        <IconRightArrow className="h-5 w-5 text-neutral-900 dark:text-neutral-100" />
      </button> */}
      <div className="flex h-auto flex-wrap items-center justify-start gap-4 rounded-[5rem] bg-white dark:bg-transparent">
        {items.map((item, index) => (
          <React.Fragment key={index}>
            <Link href={item.link}>
              <span className="rounded py-1 pr-2 font-neue text-xl font-medium leading-9 tracking-tight text-[#272727] transition-colors duration-200 hover:text-gray-400 dark:text-neutral-100 dark:hover:text-gray-500">
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
};
