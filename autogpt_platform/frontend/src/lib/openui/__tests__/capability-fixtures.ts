export const salesColumns = [
  { key: "month", label: "Month", type: "text", required: true },
  { key: "revenue", label: "Revenue", type: "number", unit: "USD" },
  { key: "cost", label: "Cost", type: "number", unit: "USD" },
];
export const salesRows = [
  { id: "jan", cells: { month: "January", revenue: 100, cost: 40 } },
  { id: "feb", cells: { month: "February", revenue: 200, cost: null } },
];
export function dataDashboard(rows = salesRows) {
  return `root = Workspace("Sales review", "Supplied data", [form, dashboard])
form = Form("sales", "Edit observations", [grid], "Review changes", "Review these observations")
grid = DataGrid("data", "Sales data", ${JSON.stringify(salesColumns)}, ${JSON.stringify(rows)}, true, "multiple", [], true)
dashboard = Dashboard("sales_dashboard", "Sales overview", ${JSON.stringify(salesColumns)}, [], [{title: "Revenue and cost", kind: "line", x: "month", series: [{key: "revenue", label: "Revenue"}, {key: "cost", label: "Cost"}], unit: "USD"}], [{label: "Total revenue", column: "revenue", aggregate: "sum", unit: "USD"}, {label: "Total cost", column: "cost", aggregate: "sum", unit: "USD"}], {form: "sales", field: "data"})`;
}

export function formView(control: string, extras = "") {
  return `root = Workspace("Review", "Supplied choices", [form])\nform = Form("review", "Your choices", [control${extras ? ", extra" : ""}], "Continue", "Use my choices")\ncontrol = ${control}${extras ? `\nextra = ${extras}` : ""}`;
}

export const tripProps = {
  name: "trip",
  title: "SF day",
  days: [
    {
      label: "Saturday",
      date: "2026-10-17",
      windows: [{ start: "09:00", end: "17:00" }],
    },
  ],
  tasks: [
    {
      id: "bridge_visit",
      title: "Golden Gate Bridge",
      placeId: "bridge",
      minutes: 60,
      done: false,
      splittable: false,
    },
    {
      id: "museum_visit",
      title: "Museum visit",
      placeId: "museum",
      minutes: 90,
      done: false,
      splittable: false,
    },
  ],
  timeZone: "America/Los_Angeles",
  locations: [
    {
      id: "bridge",
      name: "Golden Gate Bridge",
      latitude: 37.8199,
      longitude: -122.4783,
      category: "Landmark",
      detail: "Fixture location supplied for testing.",
    },
    {
      id: "museum",
      name: "Museum",
      latitude: 37.7857,
      longitude: -122.4011,
      category: "Museum",
      detail: "Fixture location supplied for testing.",
    },
  ],
  legs: [
    {
      from: "bridge",
      to: "museum",
      minutes: 45,
      mode: "Transit estimate",
      validOn: "2026-10-17",
    },
  ],
};
export const tripSource = formView(
  `Itinerary(${[tripProps.name, tripProps.title, tripProps.days, tripProps.tasks, tripProps.timeZone, tripProps.locations, tripProps.legs].map((arg) => JSON.stringify(arg)).join(", ")})`,
);
