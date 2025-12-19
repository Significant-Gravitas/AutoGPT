"use client";

import { useState } from "react";
import type { LlmModelMigration } from "@/lib/autogpt-server-api/types";
import { Button } from "@/components/atoms/Button/Button";
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
      <table className="w-full">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="px-4 py-3 text-left text-sm font-medium">
              Migration
            </th>
            <th className="px-4 py-3 text-left text-sm font-medium">Reason</th>
            <th className="px-4 py-3 text-left text-sm font-medium">
              Nodes Affected
            </th>
            <th className="px-4 py-3 text-left text-sm font-medium">
              Custom Cost
            </th>
            <th className="px-4 py-3 text-left text-sm font-medium">Created</th>
            <th className="px-4 py-3 text-right text-sm font-medium">
              Actions
            </th>
          </tr>
        </thead>
        <tbody>
          {migrations.map((migration) => (
            <MigrationRow key={migration.id} migration={migration} />
          ))}
        </tbody>
      </table>
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
        err instanceof Error ? err.message : "Failed to revert migration"
      );
    } finally {
      setIsReverting(false);
    }
  }

  const createdDate = new Date(migration.created_at);

  return (
    <>
      <tr className="border-b last:border-0">
        <td className="px-4 py-3">
          <div className="text-sm">
            <span className="font-medium">{migration.source_model_slug}</span>
            <span className="mx-2 text-muted-foreground">→</span>
            <span className="font-medium">{migration.target_model_slug}</span>
          </div>
        </td>
        <td className="px-4 py-3">
          <div className="text-sm text-muted-foreground">
            {migration.reason || "—"}
          </div>
        </td>
        <td className="px-4 py-3">
          <div className="text-sm">{migration.node_count}</div>
        </td>
        <td className="px-4 py-3">
          <div className="text-sm">
            {migration.custom_credit_cost !== null
              ? `${migration.custom_credit_cost} credits`
              : "—"}
          </div>
        </td>
        <td className="px-4 py-3">
          <div className="text-sm text-muted-foreground">
            {createdDate.toLocaleDateString()}{" "}
            {createdDate.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        </td>
        <td className="px-4 py-3 text-right">
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
        </td>
      </tr>
      {error && (
        <tr>
          <td colSpan={6} className="px-4 py-2">
            <div className="rounded border border-red-200 bg-red-50 p-2 text-sm text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
              {error}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}
