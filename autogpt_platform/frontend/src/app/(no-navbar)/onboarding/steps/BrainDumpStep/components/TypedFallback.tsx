"use client";

import { Button } from "@/components/atoms/Button/Button";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Input } from "@/components/atoms/Input/Input";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

export function TypedFallback({ value, onChange, onSubmit }: Props) {
  return (
    <FadeIn className="w-full">
      <div className="flex w-full flex-col items-center gap-3">
        <Input
          id="brain-dump-text"
          type="textarea"
          label="Your brain dump"
          hideLabel
          rows={6}
          className="rounded-xl border-zinc-200"
          placeholder="What repeats every week? What would you hand off first?"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          autoFocus
        />
        <Button
          size="small"
          onClick={onSubmit}
          disabled={!value.trim()}
          className="h-10 w-56 rounded-xl"
        >
          I&apos;m done
        </Button>
      </div>
    </FadeIn>
  );
}
