import type { LlmRouteResponse } from "@/app/api/__generated__/models/llmRouteResponse";
import type { RouteWarning } from "@/app/api/__generated__/models/routeWarning";
import { Badge } from "@/components/atoms/Badge/Badge";
import { SimpleTable } from "../SimpleTable";

interface Props {
  routes: LlmRouteResponse[];
  warnings: RouteWarning[];
}

const MODES = ["fast", "thinking"] as const;
const TIERS = ["standard", "advanced"] as const;

export function RoutingPanel({ routes, warnings }: Props) {
  const cellFor = (mode: string, tier: string) =>
    routes.find(
      (r) => r.surface === "copilot" && r.mode === mode && r.tier === tier,
    );

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-muted-foreground">
        Admin-set routing cells. Resolution order: LaunchDarkly override → these
        cells → env defaults. Empty cells fall through.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-muted-foreground">
              <th className="px-3 py-2 font-medium">Mode</th>
              {TIERS.map((tier) => (
                <th key={tier} className="px-3 py-2 font-medium capitalize">
                  {tier}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MODES.map((mode) => (
              <tr key={mode} className="border-b last:border-0">
                <td className="px-3 py-2 font-medium capitalize">{mode}</td>
                {TIERS.map((tier) => {
                  const cell = cellFor(mode, tier);
                  return (
                    <td key={tier} className="px-3 py-2">
                      {cell ? (
                        <code className="text-xs">{cell.model_slug}</code>
                      ) : (
                        <span className="text-muted-foreground">
                          — (falls through)
                        </span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <h3 className="mb-2 text-sm font-semibold">
          Resolution warnings (last 7 days)
        </h3>
        <SimpleTable
          columns={["Slug", "Reason", "Layer", "Count", "Last seen"]}
          rows={warnings.map((w) => [
            <code key="slug" className="text-xs">
              {w.slug}
            </code>,
            w.reason,
            <Badge key="layer" variant="info" size="small">
              {w.last_layer}
            </Badge>,
            String(w.count),
            new Date(w.last_seen).toLocaleString(),
          ])}
          emptyLabel="No routing refusals recorded — LaunchDarkly and cell slugs all resolve"
        />
      </div>
    </div>
  );
}
