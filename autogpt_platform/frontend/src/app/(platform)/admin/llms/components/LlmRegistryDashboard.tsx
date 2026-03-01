"use client";

import type { LlmModel } from "@/app/api/__generated__/models/llmModel";
import type { LlmModelCreator } from "@/app/api/__generated__/models/llmModelCreator";
import type { LlmModelMigration } from "@/app/api/__generated__/models/llmModelMigration";
import type { LlmProvider } from "@/app/api/__generated__/models/llmProvider";
import { ErrorBoundary } from "@/components/molecules/ErrorBoundary/ErrorBoundary";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { AddProviderModal } from "./AddProviderModal";
import { AddModelModal } from "./AddModelModal";
import { AddCreatorModal } from "./AddCreatorModal";
import { ProviderList } from "./ProviderList";
import { ModelsTable } from "./ModelsTable";
import { MigrationsTable } from "./MigrationsTable";
import { CreatorsTable } from "./CreatorsTable";
import { RecommendedModelSelector } from "./RecommendedModelSelector";

interface Props {
  providers: LlmProvider[];
  models: LlmModel[];
  migrations: LlmModelMigration[];
  creators: LlmModelCreator[];
}

function AdminErrorFallback() {
  return (
    <div className="mx-auto max-w-xl p-6">
      <ErrorCard
        responseError={{
          message:
            "An error occurred while loading the LLM Registry. Please refresh the page.",
        }}
        context="llm-registry"
        onRetry={() => window.location.reload()}
      />
    </div>
  );
}

export function LlmRegistryDashboard({
  providers,
  models,
  migrations,
  creators,
}: Props) {
  return (
    <ErrorBoundary fallback={<AdminErrorFallback />} context="llm-registry">
      <div className="mx-auto p-6">
        <div className="flex flex-col gap-6">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold">LLM Registry</h1>
            <p className="text-muted-foreground">
              Manage providers, creators, models, and credit pricing
            </p>
          </div>

          {/* Active Migrations Section - Only show if there are migrations */}
          {migrations.length > 0 && (
            <div className="rounded-lg border border-primary/30 bg-primary/5 p-6 shadow-sm">
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

          {/* Providers & Creators Section - Side by Side */}
          <div className="grid gap-6 lg:grid-cols-2">
            {/* Providers */}
            <div className="rounded-lg border bg-card p-6 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold">Providers</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Who hosts/serves the models
                  </p>
                </div>
                <AddProviderModal />
              </div>
              <ProviderList providers={providers} />
            </div>

            {/* Creators */}
            <div className="rounded-lg border bg-card p-6 shadow-sm">
              <div className="mb-4 flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold">Creators</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Who made/trained the models
                  </p>
                </div>
                <AddCreatorModal />
              </div>
              <CreatorsTable creators={creators} />
            </div>
          </div>

          {/* Models Section */}
          <div className="rounded-lg border bg-card p-6 shadow-sm">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h2 className="text-xl font-semibold">Models</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                  Toggle availability, adjust context windows, and update credit
                  pricing
                </p>
              </div>
              <AddModelModal providers={providers} creators={creators} />
            </div>

            {/* Recommended Model Selector */}
            <div className="mb-6">
              <RecommendedModelSelector models={models} />
            </div>

            <ModelsTable
              models={models}
              providers={providers}
              creators={creators}
            />
          </div>
        </div>
      </div>
    </ErrorBoundary>
  );
}
