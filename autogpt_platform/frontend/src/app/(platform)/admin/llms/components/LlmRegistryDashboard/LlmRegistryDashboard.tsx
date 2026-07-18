"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { SourceBadge } from "../RegistryBadges";
import { RoutingPanel } from "../RoutingPanel/RoutingPanel";
import { SimpleTable } from "../SimpleTable";
import { CreatorsSection } from "./components/CreatorsSection";
import { MigrationsSection } from "./components/MigrationsSection";
import { ModelsSection } from "./components/ModelsSection";
import { useLlmRegistryDashboard } from "./useLlmRegistryDashboard";

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border p-4">
      <h2 className="mb-3 text-xl font-semibold">{title}</h2>
      {children}
    </section>
  );
}

function SectionLoader() {
  return (
    <div className="flex flex-col gap-2">
      <Skeleton className="h-6 w-full" />
      <Skeleton className="h-6 w-full" />
      <Skeleton className="h-6 w-2/3" />
    </div>
  );
}

export function LlmRegistryDashboard() {
  const { models, providers, creators, migrations, routes, routeWarnings } =
    useLlmRegistryDashboard();

  const modelList = models.data?.models ?? [];
  const providerList = providers.data?.providers ?? [];
  const creatorList = creators.data?.creators ?? [];

  return (
    <div className="flex flex-col gap-6">
      <Section title="Copilot Routing">
        {routes.isLoading || routeWarnings.isLoading || models.isLoading ? (
          <SectionLoader />
        ) : routes.isError || routeWarnings.isError ? (
          <ErrorCard
            isSuccess={false}
            context="copilot routing"
            httpError={{ status: 500 }}
          />
        ) : (
          <RoutingPanel
            routes={routes.data?.routes ?? []}
            warnings={routeWarnings.data ?? []}
            models={modelList}
          />
        )}
      </Section>

      <Section title="Models">
        {models.isLoading || providers.isLoading || creators.isLoading ? (
          <SectionLoader />
        ) : models.isError ? (
          <ErrorCard
            isSuccess={false}
            context="models"
            httpError={{ status: 500 }}
          />
        ) : (
          <ModelsSection
            models={modelList}
            providers={providerList}
            creators={creatorList}
          />
        )}
      </Section>

      <Section title="Providers">
        {providers.isLoading ? (
          <SectionLoader />
        ) : providers.isError ? (
          <ErrorCard
            isSuccess={false}
            context="providers"
            httpError={{ status: 500 }}
          />
        ) : (
          <SimpleTable
            columns={["Name", "Display Name", "Models", "Source"]}
            rows={providerList.map((p) => [
              <code key="name" className="text-xs">
                {p.name}
              </code>,
              p.display_name,
              String(p.model_count ?? 0),
              <SourceBadge key="src" source={p.source ?? "SEED"} />,
            ])}
          />
        )}
      </Section>

      <Section title="Creators">
        {creators.isLoading ? (
          <SectionLoader />
        ) : creators.isError ? (
          <ErrorCard
            isSuccess={false}
            context="creators"
            httpError={{ status: 500 }}
          />
        ) : (
          <CreatorsSection creators={creatorList} />
        )}
      </Section>

      <Section title="Migrations">
        {migrations.isLoading ? (
          <SectionLoader />
        ) : migrations.isError ? (
          <ErrorCard
            isSuccess={false}
            context="migrations"
            httpError={{ status: 500 }}
          />
        ) : (
          <MigrationsSection migrations={migrations.data?.migrations ?? []} />
        )}
      </Section>
    </div>
  );
}
