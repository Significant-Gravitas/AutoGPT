"use client";

import { Button } from "@/components/atoms/Button/Button";
import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  MAX_TYPED_CHARS,
  TYPED_TOO_LONG_COPY,
  typedCharsLeftHint,
} from "../helpers";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  isTooLong: boolean;
  rows?: number;
}

const LIMIT_HINT_ID = "brain-dump-text-limit";

export function TypedFallback({
  value,
  onChange,
  onSubmit,
  isTooLong,
  rows = 6,
}: Props) {
  return (
    <FadeIn className="w-full">
      <div className="flex w-full flex-col items-center gap-3">
        <div className="flex w-full flex-col gap-1">
          <Input
            id="brain-dump-text"
            type="textarea"
            label="Your brain dump"
            hideLabel
            rows={rows}
            className="rounded-xl border-border-base"
            placeholder="What repeats every week? What would you hand off first?"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            maxLength={MAX_TYPED_CHARS}
            aria-describedby={LIMIT_HINT_ID}
            wrapperClassName="mb-0"
            autoFocus
          />
          {isTooLong ? (
            <Text
              variant="small"
              tone="danger"
              role="alert"
              id={LIMIT_HINT_ID}
              className="text-left"
            >
              {TYPED_TOO_LONG_COPY}
            </Text>
          ) : (
            <Text
              variant="small"
              tone="muted"
              id={LIMIT_HINT_ID}
              className="min-h-[1.125rem] text-right"
            >
              {typedCharsLeftHint(value.length)}
            </Text>
          )}
        </div>
        <Button
          size="small"
          onClick={onSubmit}
          disabled={!value.trim()}
          className="h-10 w-56"
        >
          I&apos;m done
        </Button>
      </div>
    </FadeIn>
  );
}
