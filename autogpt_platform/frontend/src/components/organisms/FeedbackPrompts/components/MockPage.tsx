import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";

interface Props {
  title: string;
  children?: ReactNode;
  corner?: ReactNode;
  centered?: boolean;
}

const NAV = ["Home", "Tasks", "Library", "Marketplace"];

export function MockPage({ title, children, corner, centered }: Props) {
  return (
    <div className="relative min-h-screen bg-background">
      <header className="flex h-14 items-center gap-6 border-b border-border-base bg-surface px-6">
        <Text variant="body-medium" className="!text-foreground-strong">
          AutoGPT
        </Text>
        {NAV.map((item) => (
          <Text key={item} variant="body" className="!text-foreground-tertiary">
            {item}
          </Text>
        ))}
        <div className="ml-auto size-8 rounded-full bg-surface-active" />
      </header>
      <main
        className={
          centered
            ? "flex min-h-[calc(100vh-3.5rem)] flex-col items-center justify-center gap-6 px-6"
            : "mx-auto flex max-w-4xl flex-col gap-6 px-6 py-10"
        }
      >
        {centered ? null : <Text variant="h3">{title}</Text>}
        {children ?? <SkeletonBlocks />}
      </main>
      {corner ? (
        <div className="fixed bottom-6 right-6 z-40">{corner}</div>
      ) : null}
    </div>
  );
}

export function SkeletonBlocks() {
  return (
    <div className="flex flex-col gap-4">
      <div className="h-36 rounded-2xl border border-border-base bg-surface" />
      <div className="grid grid-cols-3 gap-4">
        <div className="h-24 rounded-2xl border border-border-base bg-surface" />
        <div className="h-24 rounded-2xl border border-border-base bg-surface" />
        <div className="h-24 rounded-2xl border border-border-base bg-surface" />
      </div>
      <div className="h-48 rounded-2xl border border-border-base bg-surface" />
    </div>
  );
}
