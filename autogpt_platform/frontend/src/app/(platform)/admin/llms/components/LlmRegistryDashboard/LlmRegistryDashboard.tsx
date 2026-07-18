"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { EnabledBadge, SourceBadge, VisibilityBadge } from "../RegistryBadges";
import { RoutingPanel } from "../RoutingPanel/RoutingPanel";
import { SimpleTable } from "../SimpleTable";
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

  return (
    <div className="flex flex-col gap-6">
      <Section title="Copilot Routing">
        {routes.isLoading || routeWarnings.isLoading ? (
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
          />
        )}
      </Section>

      <Section title="Models">
        {models.isLoading ? (
          <SectionLoader />
        ) : models.isError ? (
          <ErrorCard
            isSuccess={false}
            context="models"
            httpError={{ status: 500 }}
          />
        ) : (
          <SimpleTable
            columns={[
              "Slug",
              "Name",
              "Creator",
              "Context",
              "Tier",
              "Status",
              "Visibility",
              "Source",
              "Recommended",
            ]}
            rows={(models.data?.models ?? []).map((m) => [
              <code key="slug" className="text-xs">
                {m.slug}
              </code>,
              m.display_name,
              m.creator?.display_name ?? "—",
              m.context_window.toLocaleString(),
              String(m.price_tier),
              <EnabledBadge key="enabled" enabled={m.is_enabled} />,
              <VisibilityBadge key="vis" visibility={m.visibility} />,
              <SourceBadge key="src" source={m.source} />,
              m.is_recommended ? "★" : "",
            ])}
            emptyLabel="No models in the registry yet — the bundled catalog imports on backend startup"
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
            rows={(providers.data?.providers ?? []).map((p) => [
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
          <SimpleTable
            columns={["Name", "Display Name", "Website", "Source"]}
            rows={(creators.data?.creators ?? []).map((c) => [
              <code key="name" className="text-xs">
                {c.name}
              </code>,
              c.display_name,
              c.website_url ?? "—",
              <SourceBadge key="src" source={c.source ?? "SEED"} />,
            ])}
          />
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
          <SimpleTable
            columns={["From", "To", "Nodes", "Reason", "Reverted", "Created"]}
            rows={(migrations.data?.migrations ?? []).map((mig) => [
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
            ])}
            emptyLabel="No model migrations recorded"
          />
        )}
      </Section>
    </div>
  );
}
