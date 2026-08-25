interface FreightAccount {
  id: string;
  name: string;
}

interface Arrival {
  job_id: string;
  mode: "sea" | "rail";
  origin: string;
  destination: string;
  eta: string;
  customer: string;
  status: string;
}

interface FreightException {
  job_id: string;
  severity: "high" | "medium";
  summary: string;
  owner: string;
}

interface TenantData {
  account: FreightAccount;
  summary: {
    as_of: string;
    active_jobs: number;
    arriving_this_week: number;
    exceptions_open: number;
    shipments_at_risk: number;
    revenue_mtd_gbp: number;
    gross_profit_mtd_gbp: number;
    on_time_pct: number;
    top_trade_lane: string;
    top_exception: string;
  };
  arrivals: Arrival[];
  exceptions: FreightException[];
}

const TENANTS: Record<string, TenantData> = {
  "fd-account-77": {
    account: { id: "fd-account-77", name: "Northstar Freight" },
    summary: {
      as_of: "2026-08-24T16:00:00Z",
      active_jobs: 148,
      arriving_this_week: 23,
      exceptions_open: 7,
      shipments_at_risk: 3,
      revenue_mtd_gbp: 412_850,
      gross_profit_mtd_gbp: 64_720,
      on_time_pct: 94.1,
      top_trade_lane: "Shanghai to Felixstowe",
      top_exception: "NSF-1042: customs entry missing for vessel Ever Atlas",
    },
    arrivals: [
      {
        job_id: "NSF-1042",
        mode: "sea",
        origin: "Shanghai",
        destination: "Felixstowe",
        eta: "2026-08-26",
        customer: "Aster Components",
        status: "customs hold",
      },
      {
        job_id: "NSF-1078",
        mode: "sea",
        origin: "Ningbo",
        destination: "Southampton",
        eta: "2026-08-27",
        customer: "Beacon Retail",
        status: "documents complete",
      },
      {
        job_id: "NSF-1091",
        mode: "rail",
        origin: "Duisburg",
        destination: "Birmingham",
        eta: "2026-08-28",
        customer: "Crown Industrial",
        status: "on schedule",
      },
    ],
    exceptions: [
      {
        job_id: "NSF-1042",
        severity: "high",
        summary: "Customs entry is missing before vessel discharge.",
        owner: "Import Operations",
      },
      {
        job_id: "NSF-1059",
        severity: "high",
        summary: "Original bill of lading has not been surrendered.",
        owner: "Documentation",
      },
      {
        job_id: "NSF-1066",
        severity: "medium",
        summary: "Haulier collection slot is not confirmed.",
        owner: "Transport",
      },
    ],
  },
  "fd-account-88": {
    account: { id: "fd-account-88", name: "Harbour & Rail Logistics" },
    summary: {
      as_of: "2026-08-24T16:00:00Z",
      active_jobs: 61,
      arriving_this_week: 9,
      exceptions_open: 2,
      shipments_at_risk: 1,
      revenue_mtd_gbp: 187_400,
      gross_profit_mtd_gbp: 31_220,
      on_time_pct: 97.8,
      top_trade_lane: "Rotterdam to Immingham",
      top_exception: "HBR-2208: final-mile rail slot awaits confirmation",
    },
    arrivals: [
      {
        job_id: "HBR-2208",
        mode: "rail",
        origin: "Rotterdam",
        destination: "Immingham",
        eta: "2026-08-25",
        customer: "Delta Steelworks",
        status: "rail slot pending",
      },
      {
        job_id: "HBR-2231",
        mode: "sea",
        origin: "Antwerp",
        destination: "Hull",
        eta: "2026-08-29",
        customer: "Elder Foods",
        status: "on schedule",
      },
    ],
    exceptions: [
      {
        job_id: "HBR-2208",
        severity: "high",
        summary: "Final-mile rail slot awaits terminal confirmation.",
        owner: "Rail Operations",
      },
      {
        job_id: "HBR-2214",
        severity: "medium",
        summary: "Customer delivery window needs reconfirmation.",
        owner: "Customer Service",
      },
    ],
  },
};

export type ReportName =
  | "get_operations_summary"
  | "list_arrivals"
  | "list_exceptions";

export function tenantExists(externalAccountID: string): boolean {
  return externalAccountID in TENANTS;
}

export function report(externalAccountID: string, name: ReportName): unknown {
  const tenant = TENANTS[externalAccountID];
  if (!tenant) return undefined;
  if (name === "get_operations_summary") {
    return { account: tenant.account, ...tenant.summary };
  }
  if (name === "list_arrivals") {
    return { account: tenant.account, arrivals: tenant.arrivals };
  }
  return { account: tenant.account, exceptions: tenant.exceptions };
}
