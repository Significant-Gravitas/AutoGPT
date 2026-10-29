import React from "react";

interface CardWrapperProps {
  children: React.ReactNode;
  className?: string;
}

export function CardWrapper({ children, className = "" }: CardWrapperProps) {
  return (
    <div
      data-slot="error-card"
      className={`relative my-6 overflow-hidden rounded-xl ${className}`}
    >
      {/* Gradient border */}
      <div
        data-slot="error-card-border"
        className="bg-linear-135 absolute inset-0 rounded-xl from-surface-hover via-surface-active to-surface-hover p-px"
      >
        <div
          data-slot="error-card-surface"
          className="h-full w-full rounded-xl bg-surface"
        />
      </div>
      {children}
    </div>
  );
}
