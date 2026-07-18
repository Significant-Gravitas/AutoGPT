"use client";

import type { LlmMigrationAdminResponse } from "@/app/api/__generated__/models/llmMigrationAdminResponse";
import { Button } from "@/components/atoms/Button/Button";
import { ArrowCounterClockwiseIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { RevertMigrationDialog } from "../../RevertMigrationDialog/RevertMigrationDialog";
import { SimpleTable } from "../../SimpleTable";

interface Props {
  migrations: LlmMigrationAdminResponse[];
}

export function MigrationsSection({ migrations }: Props) {
  const [reverting, setReverting] = useState<LlmMigrationAdminResponse | null>(
    null,
  );

  return (
    <div className="flex flex-col gap-3">
      <SimpleTable
        columns={[
          "From",
          "To",
          "Nodes",
          "Reason",
          "Reverted",
          "Created",
          "Actions",
        ]}
        rows={migrations.map((mig) => [
          <code key="from" className="text-xs">
            {mig.source_model_slug}
          </code>,
          <code key="to" className="text-xs">
            {mig.target_model_slug}
          </code>,
          String(mig.node_count),
          mig.reason ?? "—",
          mig.is_reverted ? "yes" : "no",
          new Date(mig.created_at).toLocaleString(),
          mig.is_reverted ? (
            ""
          ) : (
            <Button
              key="revert"
              size="icon"
              variant="icon"
              aria-label={`Revert migration from ${mig.source_model_slug}`}
              onClick={() => setReverting(mig)}
            >
              <ArrowCounterClockwiseIcon size={16} />
            </Button>
          ),
        ])}
        emptyLabel="No model migrations recorded"
      />
      {reverting && (
        <RevertMigrationDialog
          open
          migration={reverting}
          onClose={() => setReverting(null)}
        />
      )}
    </div>
  );
}
