import type { PageNotice } from "@/app/api/__generated__/models/pageNotice";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { PageEditing } from "../Page";

interface Props {
  section: PageNotice;
  editing?: PageEditing;
}

export function NoticeSection({ section, editing }: Props) {
  return (
    <div
      role="status"
      className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-warning-surface p-4"
    >
      <Text variant="body">{section.message}</Text>
      {section.reason === "stale" && editing ? (
        <Button
          variant="outline"
          size="small"
          onClick={() => editing.onChangeWithOtto(section.index, null)}
        >
          Ask Otto to fix it
        </Button>
      ) : null}
    </div>
  );
}
