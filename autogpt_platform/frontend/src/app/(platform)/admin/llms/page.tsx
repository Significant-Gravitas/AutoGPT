import { withRoleAccess } from "@/lib/withRoleAccess";
import {
  fetchLlmModels,
  fetchLlmProviders,
} from "./actions";
import { AddProviderForm } from "./components/AddProviderForm";
import { AddModelForm } from "./components/AddModelForm";
import { ProviderList } from "./components/ProviderList";
import { ModelsTable } from "./components/ModelsTable";

async function LlmRegistryDashboard() {
  const [providersResponse, modelsResponse] = await Promise.all([
    fetchLlmProviders(),
    fetchLlmModels(),
  ]);

  const providers = providersResponse.providers;
  const models = modelsResponse.models;

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-12 p-8">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">LLM Registry</h1>
        <p className="text-base text-muted-foreground">
          Manage supported providers, models, and credit pricing
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <AddProviderForm />
        <AddModelForm providers={providers} />
      </div>

      <div className="space-y-6">
        <div className="space-y-2">
          <h2 className="text-3xl font-semibold tracking-tight">Providers</h2>
          <p className="text-sm text-muted-foreground">
            Default credentials and feature flags for upstream vendors.
          </p>
        </div>
        <ProviderList providers={providers} />
      </div>

      <div className="space-y-6">
        <div className="space-y-2">
          <h2 className="text-3xl font-semibold tracking-tight">Models</h2>
          <p className="text-sm text-muted-foreground">
            Toggle availability, adjust context windows, and update credit pricing.
          </p>
        </div>
        <ModelsTable models={models} providers={providers} />
      </div>
    </div>
  );
}

export default async function AdminLlmRegistryPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedDashboard = await withAdminAccess(LlmRegistryDashboard);
  return <ProtectedDashboard />;
}

