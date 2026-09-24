import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import { cn } from "@/lib/utils";
import { COPILOT_TOOL_CATALOG } from "../../ToolChain/toolCatalog";
import { RowIcon } from "../../ToolChain/RowIcon";
import { type ApprovalItem, isHeldRead } from "../helpers";

interface Props {
  item: ApprovalItem;
  compact?: boolean;
}

export function ApprovalHeadline({ item, compact = false }: Props) {
  const category =
    COPILOT_TOOL_CATALOG[item.toolName]?.category ??
    (isHeldRead(item) ? "web" : "other");
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
        as="h3"
        className={cn(
          "min-w-0 flex-1 pt-[3px] text-zinc-900",
          compact ? "truncate" : "text-pretty",
        )}
      >
        <HeadlineText item={item} />
      </Text>
      {item.subject.irreversible && (
        <span className="mt-0.5 shrink-0 rounded-md bg-red-50 px-1.5 py-0.5 text-xs font-medium text-red-700">
          Can&apos;t be undone
        </span>
      )}
    </div>
  );
}

export function HeadlineText({ item }: Props) {
  if (isHeldRead(item)) {
    return (
      <>
        Let {AUTOPILOT_NAME} read{" "}
        <b className="font-semibold" translate="no">
          {item.source ?? item.subject.name}
        </b>
        ?
      </>
    );
  }
  const { ask, object } = item.headline;
  return (
    <>
      {ask}
      {object && (
        <>
          {" "}
          <b className="font-semibold" translate="no">
            {object}
          </b>
        </>
      )}
    </>
  );
}
