"use client";

import type {
  LlmModel,
  LlmModelMigration,
  LlmProvider,
} from "@/lib/autogpt-server-api/types";
import { AddProviderModal } from "./AddProviderModal";
import { AddModelModal } from "./AddModelModal";
import { ProviderList } from "./ProviderList";
import { ModelsTable } from "./ModelsTable";
import { MigrationsTable } from "./MigrationsTable";

interface Props {
  providers: LlmProvider[];
  models: LlmModel[];
  migrations: LlmModelMigration[];
}

export function LlmRegistryDashboard({ providers, models, migrations }: Props) {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">LLM Registry</h1>
            <p className="text-gray-500">
              Manage supported providers, models, and credit pricing
            </p>
          </div>
          <div className="flex gap-2">
            <AddModelModal providers={providers} />
            <AddProviderModal />
          </div>
        </div>

        {/* Active Migrations Section - Only show if there are migrations */}
        {migrations.length > 0 && (
          <div className="rounded-lg border border-blue-200 bg-blue-50 p-6 shadow-sm dark:border-blue-900 dark:bg-blue-950">
            <div className="mb-4">
              <h2 className="text-xl font-semibold">Active Migrations</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                These migrations can be reverted to restore workflows to their
                original model
              </p>
            </div>
            <MigrationsTable migrations={migrations} />
          </div>
        )}

        {/* Providers Section */}
        <div className="rounded-lg border bg-white p-6 shadow-sm dark:bg-background">
          <div className="mb-4">
            <h2 className="text-xl font-semibold">Providers</h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Default credentials and feature flags for upstream vendors
            </p>
          </div>
          <ProviderList providers={providers} />
        </div>

        {/* Models Section */}
        <div className="rounded-lg border bg-white p-6 shadow-sm dark:bg-background">
          <div className="mb-4">
            <h2 className="text-xl font-semibold">Models</h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Toggle availability, adjust context windows, and update credit
              pricing
            </p>
          </div>
          <ModelsTable models={models} providers={providers} />
        </div>
      </div>
    </div>
  );
}
