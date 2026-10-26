import { describe, expect, it } from "vitest";
import { cn } from "@/lib/utils";

// cn() must keep the classes tailwind-merge 2 kept: they still render as in
// Tailwind 3, so dropping one would change what an element looks like.
describe("cn() merges classes as tailwind-merge 2 did", () => {
  it.each([
    ["bg-surface-hover bg-gradient-to-b", "bg-surface-hover bg-gradient-to-b"],
    ["border-y border-2", "border-y border-2"],
    ["max-h-40 max-h-none", "max-h-40 max-h-none"],
    ["outline outline-2", "outline outline-2"],
    ["font-['Geist'] font-medium", "font-['Geist'] font-medium"],
    ["font-sans font-['Geist']", "font-['Geist']"],
    [
      "scrollbar-thumb-muted-foreground scrollbar-thumb-rounded-lg",
      "scrollbar-thumb-muted-foreground scrollbar-thumb-rounded-lg",
    ],
    ["scrollbar-thin scrollbar-none", "scrollbar-thin scrollbar-none"],
    ["text-body text-zinc-500", "text-body text-zinc-500"],
    ["text-sm text-body", "text-body"],
    ["rounded-xl rounded-2xlarge", "rounded-2xlarge"],
    ["leading-6 leading-inherit", "leading-inherit"],
    ["tracking-tight tracking-eyebrow", "tracking-eyebrow"],
    ["p-2 p-4", "p-4"],
  ])("cn(%j) is %j", (input, expected) => {
    expect(cn(input)).toBe(expected);
  });
});
