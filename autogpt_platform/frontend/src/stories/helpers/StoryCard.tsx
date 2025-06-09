import React from "react";

type StoryCardProps = {
  children: React.ReactNode;
  className?: string;
};

export function StoryCard({ children, className = "" }: StoryCardProps) {
  const themeClasses = "border-gray-200 bg-white shadow-gray-100";

  return (
    <div
      className={`rounded-lg border-2 ${themeClasses} p-8 shadow-sm ${className}`}
    >
      {children}
    </div>
  );
}
