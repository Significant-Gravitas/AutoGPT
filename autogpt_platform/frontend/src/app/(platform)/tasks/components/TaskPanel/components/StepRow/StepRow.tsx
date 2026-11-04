import type { ReactNode } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { STEP_STATUS, type StepStatus } from "../../helpers";

interface Props {
  status: StepStatus;
  label: ReactNode;
  mono?: boolean;
  trailing?: ReactNode;
}

/** One line of work: status glyph, label, and a time or live timer on the right. */
export function StepRow({ status, label, mono, trailing }: Props) {
  const glyph = status ? STEP_STATUS[status] : null;
  return (
    <span className="flex min-w-0 flex-1 items-center gap-2.5">
      <span className="flex size-4 shrink-0 items-center justify-center">
        {glyph ? (
          <Icon
            icon={glyph.icon}
            tone={glyph.tone}
            size={14}
            spin={status === "running"}
            aria-label={glyph.label}
          />
        ) : (
          <span
            aria-hidden
            className="size-1.5 rounded-full bg-foreground-faint"
          />
        )}
      </span>
      <Text
        as="span"
        variant={mono ? "small-mono" : "inbox"}
        tone={
          status === "error" ? "error" : status === "running" ? "ink" : "text"
        }
        className="min-w-0 flex-1 truncate"
      >
        {label}
      </Text>
      {trailing ? (
        <Text
          as="span"
          variant="small"
          tone="faint"
          tabular
          className="shrink-0"
        >
          {trailing}
        </Text>
      ) : null}
    </span>
  );
}
