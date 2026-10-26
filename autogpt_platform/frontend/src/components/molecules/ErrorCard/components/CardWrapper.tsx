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
      {/* Purple gradient border */}
      <div
        data-slot="error-card-border"
        className="absolute inset-0 rounded-xl bg-[linear-gradient(135deg,rgb(var(--color-surface-hover)),rgb(var(--color-surface-active)),rgb(var(--color-surface-hover)))] p-px"
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
