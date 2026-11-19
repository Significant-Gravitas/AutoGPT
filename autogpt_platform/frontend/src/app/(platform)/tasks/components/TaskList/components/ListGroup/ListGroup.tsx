import type { Expert } from "@/app/api/__generated__/models/expert";
import { NumberTicker } from "@/components/atoms/NumberTicker/NumberTicker";
import { Text } from "@/components/atoms/Text/Text";
import { AnimatePresence } from "framer-motion";
import { ownerOf, type BoardColumn } from "../../../../helpers";
import { positionLabel } from "../../../TaskBoard/boardKeys";
import { cardNav, type RovingCards } from "../../../TaskBoard/useRovingCards";
import { ListRow } from "../ListRow/ListRow";

interface Props {
  column: BoardColumn;
  experts: Expert[];
  selectedID: string | null;
  onOpen: (id: string) => void;
  roving: RovingCards;
}

/** One lane of the board as a group in the list: its name, how many, and
 *  a row per task. */
export function ListGroup({
  column,
  experts,
  selectedID,
  onOpen,
  roving,
}: Props) {
  const headingID = `task-group-${column.id}`;

  return (
    <section
      aria-labelledby={headingID}
      data-testid="list-group"
      className="min-w-0 rounded-tray bg-surface-tray p-1.5"
    >
      <header className="flex h-9 items-center gap-2 px-2.5">
        <Text as="h3" id={headingID} variant="inbox-medium" tone="ink">
          {column.label}
        </Text>
        <Text as="span" variant="inbox" tone="faint" tabular>
          <NumberTicker value={column.tasks.length} />
        </Text>
      </header>
      <ul className="grid min-w-0 gap-1">
        <AnimatePresence initial={false}>
          {column.tasks.map((task, index) => (
            <ListRow
              key={task.id}
              task={task}
              owner={experts.length > 0 ? ownerOf(task, experts) : null}
              selected={selectedID === task.id}
              onOpen={() => onOpen(task.id)}
              nav={cardNav(roving, task.id, positionLabel(column, index))}
            />
          ))}
        </AnimatePresence>
      </ul>
    </section>
  );
}
