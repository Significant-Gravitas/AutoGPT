import * as React from "react";
import Link from "next/link";

interface BreadcrumbItem {
  name: string;
  link: string;
}

interface BreadCrumbsProps {
  items: BreadcrumbItem[];
}

export const BreadCrumbs: React.FC<BreadCrumbsProps> = ({ items }) => {
  return (
    <div className="flex flex-wrap h-auto min-h-[4.375rem] items-center justify-center gap-4 rounded-[5rem] bg-white px-[1.625rem] py-[0.4375rem]">
      {items.map((item, index) => (
        <React.Fragment key={index}>
          <Link href={item.link}>
            <span className="font-['PP Neue Montreal TT'] rounded px-2 py-1 text-xl font-medium leading-9 tracking-tight text-[#272727] transition-colors duration-200 hover:text-gray-400">
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
  );
};
