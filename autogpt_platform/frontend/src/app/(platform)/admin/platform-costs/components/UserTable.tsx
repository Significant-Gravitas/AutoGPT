import type { PlatformCostDashboard } from "@/app/api/__generated__/models/platformCostDashboard";
import { formatMicrodollars, formatTokens } from "../helpers";

interface Props {
  data: PlatformCostDashboard["by_user"];
}

function UserTable({ data }: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b text-xs uppercase text-muted-foreground">
          <tr>
            <th scope="col" className="px-4 py-3">
              User
            </th>
            <th scope="col" className="px-4 py-3 text-right">
              Known Cost
            </th>
            <th scope="col" className="px-4 py-3 text-right">
              Requests
            </th>
            <th scope="col" className="px-4 py-3 text-right">
              Input Tokens
            </th>
            <th scope="col" className="px-4 py-3 text-right">
              Output Tokens
            </th>
            <th scope="col" className="px-4 py-3 text-right">
              Avg Cost / Req
            </th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr
              key={row.user_id ?? `unknown-${idx}`}
              className="border-b hover:bg-muted"
            >
              <td className="px-4 py-3">
                <div className="font-medium">{row.email || "Unknown"}</div>
                <div className="text-xs text-muted-foreground">
                  {row.user_id}
                </div>
              </td>
              <td className="px-4 py-3 text-right">
                {row.total_cost_microdollars > 0
                  ? formatMicrodollars(row.total_cost_microdollars)
                  : "-"}
              </td>
              <td className="px-4 py-3 text-right">
                {row.request_count.toLocaleString()}
              </td>
              <td className="px-4 py-3 text-right">
                {formatTokens(row.total_input_tokens)}
              </td>
              <td className="px-4 py-3 text-right">
                {formatTokens(row.total_output_tokens)}
              </td>
              <td className="px-4 py-3 text-right">
                {(row.cost_bearing_request_count ?? 0) > 0 &&
                row.total_cost_microdollars > 0
                  ? formatMicrodollars(
                      row.total_cost_microdollars /
                        (row.cost_bearing_request_count ?? 1),
                    )
                  : "-"}
              </td>
            </tr>
          ))}
          {data.length === 0 && (
            <tr>
              <td
                colSpan={6}
                className="px-4 py-8 text-center text-muted-foreground"
              >
                No cost data yet
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export { UserTable };
