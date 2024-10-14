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
    <div className="h-[4.375rem] px-[1.625rem] py-[0.4375rem] bg-white rounded-[5rem] border border-black/50 justify-center items-center gap-4 inline-flex">
      {items.map((item, index) => (
        <React.Fragment key={index}>
          <Link href={item.link}>
            <span className="text-[#272727] text-xl font-medium font-['PP Neue Montreal TT'] leading-9 tracking-tight hover:text-gray-400 transition-colors duration-200 px-2 py-1 rounded">
              {item.name}
            </span>
          </Link>
          {index < items.length - 1 && (
            <span className="text-center text-black text-2xl font-normal font-['SF Pro']">
              /
            </span>
          )}
        </React.Fragment>
      ))}
    </div>
  );
};
