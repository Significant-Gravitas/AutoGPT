import type { ReactNode } from "react";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  label: string;
  meta?: ReactNode;
  children: ReactNode;
}

/** One block of the sheet card: a quiet heading over its content. */
export function SheetSection({ label, meta, children }: Props) {
  return (
    <section aria-label={label} className="flex flex-col gap-2.5 px-5 py-4">
      <div className="flex items-center gap-2">
        <Text as="h3" variant="small" weight="medium" tone="quiet">
          {label}
        </Text>
        {meta ? (
          <Text as="span" variant="small" tone="faint" tabular>
            {meta}
          </Text>
        ) : null}
      </div>
      {children}
    </section>
  );
}
