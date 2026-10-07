import { withRoleAccess } from "@/lib/withRoleAccess";
import { BlockCostEstimatesContent } from "./components/BlockCostEstimatesContent";
import { Text } from "@/components/atoms/Text/Text";

function BlockCostEstimatesDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <Text variant="h3" as="h1">
            Block Cost Estimates
          </Text>
          <Text variant="large" tone="muted">
            Aggregate per-block average credits-per-execution from historical
            CreditTransaction USAGE rows. Use this to seed
            <code className="mx-1 text-xs">block_preflight_estimates.json</code>
            so dynamic-cost blocks charge a sensible amount up front and
            post-flight reconciliation only settles a small delta.
          </Text>
        </div>
        <BlockCostEstimatesContent />
      </div>
    </div>
  );
}

export default async function BlockCostEstimatesDashboardPage() {
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedDashboard = await withAdminAccess(BlockCostEstimatesDashboard);
  return <ProtectedDashboard />;
}
