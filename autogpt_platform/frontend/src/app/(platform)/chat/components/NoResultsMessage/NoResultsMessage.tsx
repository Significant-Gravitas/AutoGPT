import React from "react";
import { Text } from "@/components/atoms/Text/Text";
import { MagnifyingGlass, X } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

export interface NoResultsMessageProps {
  message: string;
  suggestions?: string[];
  className?: string;
}

export function NoResultsMessage({
  message,
  suggestions = [],
  className,
}: NoResultsMessageProps) {
  return (
    <div
      className={cn(
        "mx-4 my-2 flex flex-col items-center gap-4 rounded-lg border border-gray-200 bg-gray-50 p-6 dark:border-gray-800 dark:bg-gray-900",
        className,
      )}
    >
      {/* Icon */}
      <div className="relative flex h-16 w-16 items-center justify-center">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gray-200 dark:bg-gray-700">
          <MagnifyingGlass size={32} weight="bold" className="text-gray-500" />
        </div>
        <div className="absolute -right-1 -top-1 flex h-8 w-8 items-center justify-center rounded-full bg-gray-400 dark:bg-gray-600">
          <X size={20} weight="bold" className="text-white" />
        </div>
      </div>

      {/* Content */}
      <div className="text-center">
        <Text variant="h3" className="mb-2 text-gray-900 dark:text-gray-100">
          No Results Found
        </Text>
        <Text variant="body" className="text-gray-700 dark:text-gray-300">
          {message}
        </Text>
      </div>

      {/* Suggestions */}
      {suggestions.length > 0 && (
        <div className="w-full space-y-2">
          <Text
            variant="small"
            className="font-semibold text-gray-900 dark:text-gray-100"
          >
            Try these suggestions:
          </Text>
          <ul className="space-y-1 rounded-md bg-gray-100 p-4 dark:bg-gray-800">
            {suggestions.map((suggestion, index) => (
              <li
                key={index}
                className="flex items-start gap-2 text-sm text-gray-700 dark:text-gray-300"
              >
                <span className="mt-1 text-gray-500">•</span>
                <span>{suggestion}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
