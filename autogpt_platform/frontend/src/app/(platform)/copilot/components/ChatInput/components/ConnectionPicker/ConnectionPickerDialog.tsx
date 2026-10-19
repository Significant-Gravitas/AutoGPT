"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import Link from "next/link";

import { ConnectionPickerPanel } from "./ConnectionPickerPanel";
import { listedConnections } from "./helpers";
import { useConnectionPicker } from "./useConnectionPicker";

interface Props {
  /** See `ConnectionPicker`: the connection is fixed, only the tier is open. */
  connectionLocked?: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** The connection picker's sheet as a dialog, opened from the composer's +. */
export function ConnectionPickerDialog({
  connectionLocked = false,
  open,
  onOpenChange,
}: Props) {
  // Mounted while closed too: the hook also drops a lapsed Advanced tier,
  // which has to happen whether or not anyone opens the sheet.
  const picker = useConnectionPicker();

  return (
    <Dialog
      title="Model and connection"
      controlled={{ isOpen: open, set: onOpenChange }}
      styling={{ maxWidth: "26rem" }}
    >
      <Dialog.Content>
        <DialogBody picker={picker} connectionLocked={connectionLocked} />
      </Dialog.Content>
    </Dialog>
  );
}

interface BodyProps {
  picker: ReturnType<typeof useConnectionPicker>;
  connectionLocked: boolean;
}

function DialogBody({ picker, connectionLocked }: BodyProps) {
  const { offers, active, isLoading, isError } = picker;

  if (isLoading && offers.length === 0) {
    return <Skeleton className="h-32 w-full rounded-xl" />;
  }

  if (isError) {
    return (
      <p className="text-sm text-zinc-600">
        AI connections are unavailable right now. Try again in a moment.
      </p>
    );
  }

  if (offers.length === 0) {
    return (
      <p className="text-sm text-zinc-600">
        No AI connection is set up yet.{" "}
        <Link href="/settings/integrations" className="underline">
          Set one up
        </Link>
      </p>
    );
  }

  // Asked for from the menu, so the dialog always says what the chat runs
  // on: with nothing to choose, it still lists the active connection.
  const listed = listedConnections(offers, connectionLocked);
  const rows = listed.length > 0 ? listed : active ? [active] : offers;

  return <ConnectionPickerPanel picker={picker} rows={rows} />;
}
