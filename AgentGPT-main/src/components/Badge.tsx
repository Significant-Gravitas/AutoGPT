import React from "react";
import clsx from "clsx";

interface BadgeProps {
  children: React.ReactNode;
}

const Badge = ({ children }: BadgeProps) => {
  return (
    <div
      className={clsx(
        "mt-2 rounded-full bg-[#1E88E5] font-semibold text-gray-100 transition-all hover:scale-110",
        "px-1 py-1 text-xs",
        "sm:px-3 sm:py-1 sm:text-sm"
      )}
    >
      {children}
    </div>
  );
};

export default Badge;
