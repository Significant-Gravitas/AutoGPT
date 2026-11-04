import type { ReactNode } from "react";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  label: string;
  meta?: ReactNode;
  children: ReactNode;
}

/** One block of the task panel: a quiet heading over its content. */
export function PanelSection({ label, meta, children }: Props) {
  return (
    <section aria-label={label} className="flex flex-col gap-2.5 px-5 py-4">
      <div className="flex items-center gap-2">
        <Text as="h3" variant="inbox-medium" tone="quiet">
          {label}
        </Text>
        {meta ? (
          <Text as="span" variant="inbox" tone="faint" tabular>
            {meta}
          </Text>
        ) : null}
      </div>
      {children}
    </section>
  );
}
