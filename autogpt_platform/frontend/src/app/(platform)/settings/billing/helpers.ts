export const EASE_OUT = [0.16, 1, 0.3, 1] as const;

export function formatCents(cents: number): string {
  const sign = cents < 0 ? "-" : "";
  return `${sign}$${(Math.abs(cents) / 100).toFixed(2)}`;
}

export function formatRelativeReset(
  target: Date | string | undefined | null,
): { prefix: string; value: string } {
  if (!target) return { prefix: "Resets", value: "—" };
  const date = target instanceof Date ? target : new Date(target);
  if (Number.isNaN(date.getTime())) return { prefix: "Resets", value: "—" };
  const diff = date.getTime() - Date.now();
  if (diff <= 0) return { prefix: "Resets", value: "soon" };
  const hours = Math.floor(diff / (1000 * 60 * 60));
  if (hours < 24) {
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    return { prefix: "Resets in", value: `${hours}h ${minutes}m` };
  }
  return {
    prefix: "Resets",
    value: date.toLocaleString(undefined, {
      weekday: "short",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    }),
  };
}

export function formatShortDate(value: Date | string | undefined | null): string {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export interface PlanInfo {
  name: string;
  price: string;
  billingPeriod: string;
  renewsOn: string;
  status: "active" | "trial" | "canceled";
}

export interface AutopilotUsageWindow {
  label: string;
  percent: number;
  resetsPrefix: string;
  resetsValue: string;
  used: number;
  total: number;
  unit: string;
}

export interface AutopilotUsage {
  today: AutopilotUsageWindow;
  week: AutopilotUsageWindow;
}

export interface PaymentMethod {
  brand: string;
  last4: string;
  expiryMonth: string;
  expiryYear: string;
  cardholderName: string;
}

export interface Invoice {
  id: string;
  date: string;
  description: string;
  amount: string;
  status: "paid" | "pending" | "failed";
}

export const CURRENT_PLAN: PlanInfo = {
  name: "Pro",
  price: "$20",
  billingPeriod: "month",
  renewsOn: "May 28, 2026",
  status: "active",
};

export const AUTOPILOT_USAGE: AutopilotUsage = {
  today: {
    label: "Today",
    percent: 12,
    resetsPrefix: "Resets in",
    resetsValue: "21h 26m",
    used: 120,
    total: 1000,
    unit: "runs",
  },
  week: {
    label: "This Week",
    percent: 34,
    resetsPrefix: "Resets",
    resetsValue: "Sun, 5:00 PM PDT",
    used: 2380,
    total: 7000,
    unit: "runs",
  },
};

export const PAYMENT_METHOD: PaymentMethod = {
  brand: "Visa",
  last4: "4242",
  expiryMonth: "08",
  expiryYear: "2028",
  cardholderName: "Abhimanyu Yadav",
};

export interface CreditTransaction {
  id: string;
  date: string;
  description: string;
  amount: string;
  balance: string;
  kind: "credit" | "debit";
}

export const AUTOMATION_CREDITS_BALANCE = "$3.33";

export interface DailyUsage {
  day: string;
  date: string;
  amount: number;
  runs: number;
}

const DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTH_NAMES = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

const SEED_AMOUNTS = [
  0.18, 0.42, 0.95, 0.27, 0.83, 1.41, 0.36, 0.62, 1.08, 0.21, 0.55, 0.74, 0.31,
  0.49, 1.22, 0.18, 0.67, 0.93, 0.4, 0.28, 0.81, 0.34, 0.59, 1.05, 0.46, 0.72,
  0.95, 0.32, 0.65, 0.42,
];

export const DAILY_USAGE: DailyUsage[] = (() => {
  const end = new Date(2026, 3, 28); // Apr 28 2026
  return Array.from({ length: 30 }, (_, idx) => {
    const date = new Date(end);
    date.setDate(end.getDate() - (29 - idx));
    const amount = SEED_AMOUNTS[idx] ?? 0;
    return {
      day: DAY_NAMES[date.getDay() === 0 ? 6 : date.getDay() - 1],
      date: `${MONTH_NAMES[date.getMonth()]} ${date.getDate()}`,
      amount,
      runs: Math.max(1, Math.round(amount / 0.12)),
    };
  });
})();

export const CREDIT_TRANSACTIONS: CreditTransaction[] = [
  {
    id: "TXN-2026-0428-01",
    date: "Apr 28, 2026",
    description: "Agent run · Daily news digest",
    amount: "-$0.42",
    balance: "$3.33",
    kind: "debit",
  },
  {
    id: "TXN-2026-0427-02",
    date: "Apr 27, 2026",
    description: "Top up",
    amount: "+$5.00",
    balance: "$3.75",
    kind: "credit",
  },
  {
    id: "TXN-2026-0426-03",
    date: "Apr 26, 2026",
    description: "Agent run · Lead enrichment",
    amount: "-$1.18",
    balance: "-$1.25",
    kind: "debit",
  },
  {
    id: "TXN-2026-0425-04",
    date: "Apr 25, 2026",
    description: "Auto-refill",
    amount: "+$10.00",
    balance: "-$0.07",
    kind: "credit",
  },
  {
    id: "TXN-2026-0424-05",
    date: "Apr 24, 2026",
    description: "Agent run · Inbox triage",
    amount: "-$0.65",
    balance: "-$10.07",
    kind: "debit",
  },
];

export const INVOICES: Invoice[] = [
  {
    id: "INV-2026-0428",
    date: "Apr 28, 2026",
    description: "Pro plan · Monthly",
    amount: "$20.00",
    status: "paid",
  },
  {
    id: "INV-2026-0328",
    date: "Mar 28, 2026",
    description: "Pro plan · Monthly",
    amount: "$20.00",
    status: "paid",
  },
  {
    id: "INV-2026-0228",
    date: "Feb 28, 2026",
    description: "Pro plan · Monthly",
    amount: "$20.00",
    status: "paid",
  },
  {
    id: "INV-2026-0128",
    date: "Jan 28, 2026",
    description: "Pro plan · Monthly",
    amount: "$20.00",
    status: "paid",
  },
  {
    id: "INV-2025-1228",
    date: "Dec 28, 2025",
    description: "Pro plan · Monthly",
    amount: "$20.00",
    status: "paid",
  },
];
