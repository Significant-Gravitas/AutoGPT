import Link from "next/link";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { COPILOT_TOOL_CATALOG } from "../../ToolChain/toolCatalog";
import { RowIcon } from "../../ToolChain/RowIcon";
import type { ApprovalItem } from "../helpers";

interface Props {
  item: ApprovalItem;
  // Inside a button a heading is not allowed, so the line renders a span.
  compact?: boolean;
}

export function ApprovalHeadline({ item, compact = false }: Props) {
  const category = COPILOT_TOOL_CATALOG[item.toolName]?.category ?? "other";
  return (
    <div className="flex min-w-0 items-start gap-2.5">
      <span
        aria-hidden="true"
        className="flex size-7 shrink-0 items-center justify-center rounded-lg border border-zinc-200 bg-white"
      >
        <RowIcon
          row={{ key: item.reviewId, category, text: "", state: "done" }}
        />
      </span>
      <Text
        variant="body"
        as={compact ? "span" : "h3"}
        className={cn(
          "min-w-0 flex-1 pt-[3px] text-zinc-900",
          compact ? "truncate" : "text-pretty",
        )}
      >
        <HeadlineText item={item} linked={!compact} />
      </Text>
      {item.subject.irreversible && (
        <span className="mt-0.5 shrink-0 rounded-md bg-red-50 px-1.5 py-0.5 text-xs font-medium text-red-700">
          Can&apos;t be undone
        </span>
      )}
    </div>
  );
}

interface HeadlineTextProps {
  item: ApprovalItem;
  // Off wherever the line is itself a button.
  linked?: boolean;
}

export function HeadlineText({ item, linked = false }: HeadlineTextProps) {
  const { ask, object } = item.headline;
  const href = linked ? objectHref(item) : null;
  return (
    <>
      {ask}
      {object && (
        <>
          {" "}
          <b className="font-semibold" translate="no">
            {href ? (
              <Link
                href={href}
                className="underline decoration-zinc-300 underline-offset-2 hover:decoration-zinc-500 focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300"
              >
                {object}
              </Link>
            ) : (
              object
            )}
          </b>
        </>
      )}
    </>
  );
}

// The headline names a resolved id's thing, and its field is hidden, so the link lives here.
function objectHref(item: ApprovalItem) {
  const [key] = item.headlineKeys;
  return (
    item.references.find((ref) => ref.key === key && ref.href)?.href ?? null
  );
}
