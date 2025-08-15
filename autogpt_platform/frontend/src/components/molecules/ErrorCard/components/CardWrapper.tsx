import React from "react";
import { colors } from "@/components/styles/colors";

interface CardWrapperProps {
  children: React.ReactNode;
  className?: string;
}

export function CardWrapper({ children, className = "" }: CardWrapperProps) {
  return (
    <div className={`relative overflow-hidden rounded-xl ${className}`}>
      {/* Purple gradient border */}
      <div
        className="absolute inset-0 rounded-xl p-[1px]"
        style={{
          background: `linear-gradient(135deg, ${colors.zinc[500]}, ${colors.zinc[200]}, ${colors.zinc[100]})`,
        }}
      >
        <div className="h-full w-full rounded-xl bg-white" />
      </div>
      {children}
    </div>
  );
}
