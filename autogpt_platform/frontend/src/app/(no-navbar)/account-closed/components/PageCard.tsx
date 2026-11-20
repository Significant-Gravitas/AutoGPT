import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";

interface Props {
  title: string;
  // Lets a view move focus to its heading when it replaces another.
  headingID?: string;
  children: ReactNode;
}

export function PageCard({ title, headingID, children }: Props) {
  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-8">
      <div className="flex w-full max-w-md flex-col gap-4 rounded-large border border-border-base bg-surface p-6">
        <Text variant="h3" as="h1" id={headingID} tabIndex={-1}>
          {title}
        </Text>
        {children}
      </div>
    </main>
  );
}
