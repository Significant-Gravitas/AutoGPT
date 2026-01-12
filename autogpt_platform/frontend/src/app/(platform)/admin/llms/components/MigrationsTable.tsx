"use client";

import { useState } from "react";
import type { LlmModelMigration } from "@/app/api/__generated__/models/llmModelMigration";
import { Button } from "@/components/atoms/Button/Button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/atoms/Table/Table";
import { revertLlmMigrationAction } from "../actions";

export function MigrationsTable({
  migrations,
}: {
  migrations: LlmModelMigration[];
}) {
  if (!migrations.length) {
    return (
      <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
        No active migrations. Migrations are created when you disable a model
        with the &quot;Migrate existing workflows&quot; option.
      </div>
    );
  }

  return (
    <div className="rounded-lg border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Migration</TableHead>
            <TableHead>Reason</TableHead>
            <TableHead>Nodes Affected</TableHead>
            <TableHead>Custom Cost</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {migrations.map((migration) => (
            <MigrationRow key={migration.id} migration={migration} />
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function MigrationRow({ migration }: { migration: LlmModelMigration }) {
  const [isReverting, setIsReverting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleRevert(formData: FormData) {
    setIsReverting(true);
    setError(null);
    try {
      await revertLlmMigrationAction(formData);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to revert migration",
      );
    } finally {
      setIsReverting(false);
    }
  }

  const createdDate = new Date(migration.created_at);

  return (
    <>
      <TableRow>
        <TableCell>
          <div className="text-sm">
            <span className="font-medium">{migration.source_model_slug}</span>
            <span className="mx-2 text-muted-foreground">→</span>
            <span className="font-medium">{migration.target_model_slug}</span>
          </div>
        </TableCell>
        <TableCell>
          <div className="text-sm text-muted-foreground">
            {migration.reason || "—"}
          </div>
        </TableCell>
        <TableCell>
          <div className="text-sm">{migration.node_count}</div>
        </TableCell>
        <TableCell>
          <div className="text-sm">
            {migration.custom_credit_cost !== null &&
            migration.custom_credit_cost !== undefined
              ? `${migration.custom_credit_cost} credits`
              : "—"}
          </div>
        </TableCell>
        <TableCell>
          <div className="text-sm text-muted-foreground">
            {createdDate.toLocaleDateString()}{" "}
            {createdDate.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        </TableCell>
        <TableCell className="text-right">
          <form action={handleRevert} className="inline">
            <input type="hidden" name="migration_id" value={migration.id} />
            <Button
              type="submit"
              variant="outline"
              size="small"
              disabled={isReverting}
            >
              {isReverting ? "Reverting..." : "Revert"}
            </Button>
          </form>
        </TableCell>
      </TableRow>
      {error && (
        <TableRow>
          <TableCell colSpan={6}>
            <div className="rounded border border-destructive/30 bg-destructive/10 p-2 text-sm text-destructive">
              {error}
            </div>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}
