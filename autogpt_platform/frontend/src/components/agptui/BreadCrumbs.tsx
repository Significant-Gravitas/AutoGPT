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
      <button className="flex h-12 w-12 items-center justify-center rounded-full border border-neutral-200 transition-colors hover:bg-neutral-50">
        <IconLeftArrow className="h-5 w-5 text-neutral-900" />
      </button>
      <button className="flex h-12 w-12 items-center justify-center rounded-full border border-neutral-200 transition-colors hover:bg-neutral-50">
        <IconRightArrow className="h-5 w-5 text-neutral-900" />
      </button>
      <div className="flex h-auto min-h-[4.375rem] flex-wrap items-center justify-start gap-4 rounded-[5rem] bg-white">
        {items.map((item, index) => (
          <React.Fragment key={index}>
            <Link href={item.link}>
              <span className="rounded py-1 pr-2 font-neue text-xl font-medium leading-9 tracking-tight text-[#272727] transition-colors duration-200 hover:text-gray-400">
                {item.name}
              </span>
            </Link>
            {index < items.length - 1 && (
              <span className="font-['SF Pro'] text-center text-2xl font-normal text-black">
                /
              </span>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};
