import type { PageButton } from "@/app/api/__generated__/models/pageButton";
import { Button } from "@/components/atoms/Button/Button";
import type { TableRow } from "../CollectionTable/helpers";
import type { useRowButtons } from "./useRowButtons";

interface Props {
  buttons: PageButton[];
  row: TableRow;
  rowLabel: string;
  actions: ReturnType<typeof useRowButtons>;
}

export function RowButtons({ buttons, row, rowLabel, actions }: Props) {
  if (!buttons.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {buttons.map((button) => (
        <Button
          key={button.label}
          size="xs"
          variant="outline"
          aria-label={`${button.label}: ${rowLabel}`}
          loading={actions.isRunning(button, row)}
          disabled={actions.isBusy}
          onClick={() => actions.run(button, row)}
        >
          {button.label}
        </Button>
      ))}
    </div>
  );
}
