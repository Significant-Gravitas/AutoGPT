"use client";

import type { LlmCreatorAdminResponse } from "@/app/api/__generated__/models/llmCreatorAdminResponse";
import { Button } from "@/components/atoms/Button/Button";
import { PencilSimpleIcon, TrashIcon } from "@phosphor-icons/react";
import { useState } from "react";
import {
  CreatorFormDialog,
  DeleteCreatorDialog,
} from "../../CreatorDialogs/CreatorDialogs";
import { SourceBadge } from "../../RegistryBadges";
import { SimpleTable } from "../../SimpleTable";

interface Props {
  creators: LlmCreatorAdminResponse[];
}

type DialogState =
  | { kind: "closed" }
  | { kind: "create" }
  | { kind: "edit"; creator: LlmCreatorAdminResponse }
  | { kind: "delete"; creator: LlmCreatorAdminResponse };

export function CreatorsSection({ creators }: Props) {
  const [dialog, setDialog] = useState<DialogState>({ kind: "closed" });

  function close() {
    setDialog({ kind: "closed" });
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex justify-end">
        <Button size="small" onClick={() => setDialog({ kind: "create" })}>
          Add creator
        </Button>
      </div>
      <SimpleTable
        columns={["Name", "Display Name", "Website", "Source", "Actions"]}
        rows={creators.map((c) => [
          <code key="name" className="text-xs">
            {c.name}
          </code>,
          c.display_name,
          c.website_url ?? "—",
          <SourceBadge key="src" source={c.source ?? "SEED"} />,
          <div key="actions" className="flex gap-1">
            <Button
              size="icon"
              variant="icon"
              aria-label={`Edit ${c.name}`}
              onClick={() => setDialog({ kind: "edit", creator: c })}
            >
              <PencilSimpleIcon size={16} />
            </Button>
            <Button
              size="icon"
              variant="icon"
              aria-label={`Delete ${c.name}`}
              onClick={() => setDialog({ kind: "delete", creator: c })}
            >
              <TrashIcon size={16} />
            </Button>
          </div>,
        ])}
      />
      {(dialog.kind === "create" || dialog.kind === "edit") && (
        <CreatorFormDialog
          open
          editing={dialog.kind === "edit" ? dialog.creator : null}
          onClose={close}
        />
      )}
      {dialog.kind === "delete" && (
        <DeleteCreatorDialog open creator={dialog.creator} onClose={close} />
      )}
    </div>
  );
}
