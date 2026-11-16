import { describe, expect, it } from "vitest";

import { formatShortDate } from "@/app/(platform)/settings/billing/helpers";
import { SubscriptionStatusResponseTier } from "@/app/api/__generated__/models/subscriptionStatusResponseTier";

import {
  EXIT_REASON_OPTIONS,
  getTextPlaceholder,
} from "../components/ExitReasonStep/helpers";
import {
  formatAppNames,
  getReasonPanel,
} from "../components/ExitReasonStep/panel-helpers";
import {
  getPlanState,
  type AppsState,
  type DeletionSource,
  type DisconnectState,
  type ExitReason,
  type PanelAction,
  type PendingChange,
  type PlanState,
  type ReasonPanel,
} from "../helpers";

type PanelArgs = Parameters<typeof getReasonPanel>[0];

const ENDS_ON = new Date(2026, 10, 30);
const PRO_CENTS = 5000;

const PLAN_KEYS = ["max", "pro", "trial", "none", "managed", "null"] as const;
type PlanKey = (typeof PLAN_KEYS)[number];
const PLANS: Record<PlanKey, PlanState | null> = {
  max: "max",
  pro: "pro",
  trial: "trial",
  none: "none",
  managed: "managed",
  null: null,
};

const PENDING_KEYS = ["none", "cancel", "downgrade"] as const;
type PendingKey = (typeof PENDING_KEYS)[number];
const PENDING: Record<PendingKey, PendingChange | null> = {
  none: null,
  cancel: { kind: "cancel", effectiveAt: ENDS_ON },
  downgrade: { kind: "downgrade", effectiveAt: ENDS_ON },
};

const APP_NAMES = ["Gmail", "Google Calendar", "Notion", "Slack", "GitHub"];
const APPS = {
  unavailable: { status: "unavailable" },
  empty: { status: "ready", names: [] },
  one: { status: "ready", names: APP_NAMES.slice(0, 1) },
  two: { status: "ready", names: APP_NAMES.slice(0, 2) },
  three: { status: "ready", names: APP_NAMES.slice(0, 3) },
  four: { status: "ready", names: APP_NAMES.slice(0, 4) },
  five: { status: "ready", names: APP_NAMES.slice(0, 5) },
} satisfies Record<string, AppsState>;
type AppsKey = keyof typeof APPS;

const DISCONNECTS = {
  idle: { status: "idle" },
  doneOne: { status: "done", names: APP_NAMES.slice(0, 1) },
  doneTwo: { status: "done", names: APP_NAMES.slice(0, 2) },
  doneThree: { status: "done", names: APP_NAMES.slice(0, 3) },
  doneFour: { status: "done", names: APP_NAMES.slice(0, 4) },
  failedOne: { status: "failed", names: APP_NAMES.slice(0, 1) },
  failedTwo: { status: "failed", names: APP_NAMES.slice(0, 2) },
  failedThree: { status: "failed", names: APP_NAMES.slice(0, 3) },
  failedFour: { status: "failed", names: APP_NAMES.slice(0, 4) },
} satisfies Record<string, DisconnectState>;
type DisconnectKey = keyof typeof DISCONNECTS;

const ORIGINS: DeletionSource[] = ["settings", "paywall"];

const CONTEXTS = Object.entries(APPS).flatMap(([appsKey, apps]) =>
  ORIGINS.flatMap((origin) =>
    Object.entries(DISCONNECTS).map(([disconnectKey, disconnect]) => ({
      apps,
      origin,
      disconnect,
      label: `apps ${appsKey}, from ${origin}, disconnect ${disconnectKey}`,
    })),
  ),
);

const PLAN_AND_PENDING = PLAN_KEYS.flatMap((plan) =>
  PENDING_KEYS.map((pending) => ({
    planState: PLANS[plan],
    pending: PENDING[pending],
    label: `plan ${plan}, pending ${pending}`,
  })),
);

function panel(overrides: Partial<PanelArgs>) {
  return getReasonPanel({
    reason: "privacy",
    planState: null,
    pending: null,
    apps: APPS.unavailable,
    proMonthlyCents: PRO_CENTS,
    trialEndsAt: null,
    origin: "settings",
    disconnect: DISCONNECTS.idle,
    ...overrides,
  });
}

const MANAGE_APPS: PanelAction = {
  kind: "manage_apps",
  label: "Manage connected apps",
};
const DISCONNECT_ALL: PanelAction = {
  kind: "disconnect_all",
  label: "Disconnect all apps",
};
const BILLING: PanelAction = { kind: "billing", label: "Go to Billing" };
const KEEP: PanelAction = { kind: "keep_account", label: "Keep my account" };

const PLAN_ENDS: ReasonPanel = {
  copy: "Your plan already ends on Nov 30, 2026, and you won't be charged again. Your account stays after that.",
  action: KEEP,
};
const SWITCH_SCHEDULED: ReasonPanel = {
  copy: "Your switch to Pro is already scheduled for Nov 30, 2026.",
  action: null,
};
const MAX_DOWNGRADE: ReasonPanel = {
  copy: "Pro is $50.00 / month. A downgrade keeps your Experts, chats and schedules and starts when your current period ends.",
  action: BILLING,
};
const PRO_CANCEL: ReasonPanel = {
  copy: "You can cancel your plan and keep your account. You won't be charged again, and your Experts and chats stay here if you come back.",
  action: BILLING,
};
const TRIAL_ENDS: ReasonPanel = {
  copy: "Your trial already ends on Nov 30, 2026 and you won't be charged.",
  action: KEEP,
};
const TRIAL_CANCEL: ReasonPanel = {
  copy: "You can cancel your trial instead. You won't be charged.",
  action: BILLING,
};
const NO_PLAN: ReasonPanel = {
  copy: "Your account costs nothing without a plan. Your Experts and chats stay here if you come back.",
  action: KEEP,
};
const LIMITS: ReasonPanel = {
  copy: "Max gives you 8.5x the usage of Pro.",
  action: BILLING,
};
const NO_APPS: ReasonPanel = {
  copy: "You haven't connected any apps, so AutoGPT can't reach your email, calendar or files.",
  action: null,
};

function offer(list: string, action: PanelAction): ReasonPanel {
  return {
    copy: `You can disconnect ${list} without deleting your account. Our access ends as soon as you do.`,
    action,
  };
}

type MatrixRow = Record<PendingKey, ReasonPanel | null>;
type Matrix = Record<PlanKey, MatrixRow>;

function always(expected: ReasonPanel | null): MatrixRow {
  return { none: expected, cancel: expected, downgrade: expected };
}

const NOTHING: Matrix = {
  max: always(null),
  pro: always(null),
  trial: always(null),
  none: always(null),
  managed: always(null),
  null: always(null),
};

const MATRICES: Record<Exclude<ExitReason, "privacy">, Matrix> = {
  cost: {
    max: {
      none: MAX_DOWNGRADE,
      cancel: PLAN_ENDS,
      downgrade: SWITCH_SCHEDULED,
    },
    pro: { none: PRO_CANCEL, cancel: PLAN_ENDS, downgrade: PRO_CANCEL },
    trial: { none: TRIAL_CANCEL, cancel: TRIAL_ENDS, downgrade: TRIAL_CANCEL },
    none: always(NO_PLAN),
    managed: always(null),
    null: always(null),
  },
  not_using: {
    max: { none: PRO_CANCEL, cancel: PLAN_ENDS, downgrade: PRO_CANCEL },
    pro: { none: PRO_CANCEL, cancel: PLAN_ENDS, downgrade: PRO_CANCEL },
    trial: { none: TRIAL_CANCEL, cancel: TRIAL_ENDS, downgrade: TRIAL_CANCEL },
    none: always(NO_PLAN),
    managed: always(null),
    null: always(null),
  },
  missing_capability: NOTHING,
  limits: {
    max: always(null),
    pro: { none: LIMITS, cancel: null, downgrade: LIMITS },
    trial: always(null),
    none: always(null),
    managed: always(null),
    null: always(null),
  },
  other: NOTHING,
};

const NON_PRIVACY_REASONS = [
  "cost",
  "not_using",
  "missing_capability",
  "limits",
  "other",
] as const;

const MATRIX_ROWS = NON_PRIVACY_REASONS.flatMap((reason) =>
  PLAN_KEYS.flatMap((plan) =>
    PENDING_KEYS.map((pending) => ({
      reason,
      plan,
      pending,
      expected: MATRICES[reason][plan][pending],
    })),
  ),
);

const PRIVACY_IDLE_ROWS: {
  apps: AppsKey;
  origin: DeletionSource;
  expected: ReasonPanel | null;
}[] = [
  { apps: "unavailable", origin: "settings", expected: null },
  { apps: "unavailable", origin: "paywall", expected: null },
  { apps: "empty", origin: "settings", expected: NO_APPS },
  { apps: "empty", origin: "paywall", expected: NO_APPS },
  { apps: "one", origin: "settings", expected: offer("Gmail", MANAGE_APPS) },
  { apps: "one", origin: "paywall", expected: offer("Gmail", DISCONNECT_ALL) },
  {
    apps: "two",
    origin: "settings",
    expected: offer("Gmail and Google Calendar", MANAGE_APPS),
  },
  {
    apps: "two",
    origin: "paywall",
    expected: offer("Gmail and Google Calendar", DISCONNECT_ALL),
  },
  {
    apps: "three",
    origin: "settings",
    expected: offer("Gmail, Google Calendar and 1 more app", MANAGE_APPS),
  },
  {
    apps: "three",
    origin: "paywall",
    expected: offer("Gmail, Google Calendar and 1 more app", DISCONNECT_ALL),
  },
  {
    apps: "four",
    origin: "settings",
    expected: offer("Gmail, Google Calendar and 2 more apps", MANAGE_APPS),
  },
  {
    apps: "four",
    origin: "paywall",
    expected: offer("Gmail, Google Calendar and 2 more apps", DISCONNECT_ALL),
  },
  {
    apps: "five",
    origin: "settings",
    expected: offer("Gmail, Google Calendar and 3 more apps", MANAGE_APPS),
  },
  {
    apps: "five",
    origin: "paywall",
    expected: offer("Gmail, Google Calendar and 3 more apps", DISCONNECT_ALL),
  },
];

const PRIVACY_AFTER_DISCONNECT_ROWS: {
  disconnect: Exclude<DisconnectKey, "idle">;
  expected: ReasonPanel;
}[] = [
  {
    disconnect: "doneOne",
    expected: { copy: "Gmail is disconnected.", action: null },
  },
  {
    disconnect: "doneTwo",
    expected: {
      copy: "Gmail and Google Calendar are disconnected.",
      action: null,
    },
  },
  {
    disconnect: "doneThree",
    expected: {
      copy: "Gmail, Google Calendar and 1 more app are disconnected.",
      action: null,
    },
  },
  {
    disconnect: "doneFour",
    expected: {
      copy: "Gmail, Google Calendar and 2 more apps are disconnected.",
      action: null,
    },
  },
  {
    disconnect: "failedOne",
    expected: {
      copy: "Couldn't disconnect Gmail. Try again.",
      action: DISCONNECT_ALL,
    },
  },
  {
    disconnect: "failedTwo",
    expected: {
      copy: "Couldn't disconnect Gmail and Google Calendar. Try again.",
      action: DISCONNECT_ALL,
    },
  },
  {
    disconnect: "failedThree",
    expected: {
      copy: "Couldn't disconnect Gmail, Google Calendar and 1 more app. Try again.",
      action: DISCONNECT_ALL,
    },
  },
  {
    disconnect: "failedFour",
    expected: {
      copy: "Couldn't disconnect Gmail, Google Calendar and 2 more apps. Try again.",
      action: DISCONNECT_ALL,
    },
  },
];

describe("getReasonPanel", () => {
  describe("matrix for every reason except privacy", () => {
    it.each(MATRIX_ROWS)(
      "$reason · plan $plan · pending $pending",
      ({ reason, plan, pending, expected }) => {
        for (const context of CONTEXTS) {
          expect(
            panel({
              reason,
              planState: PLANS[plan],
              pending: PENDING[pending],
              apps: context.apps,
              origin: context.origin,
              disconnect: context.disconnect,
            }),
            context.label,
          ).toEqual(expected);
        }
      },
    );
  });

  describe("privacy, for every plan state and pending change", () => {
    it.each(PRIVACY_IDLE_ROWS)(
      "apps $apps · from $origin",
      ({ apps, origin, expected }) => {
        for (const { planState, pending, label } of PLAN_AND_PENDING) {
          expect(
            panel({ planState, pending, apps: APPS[apps], origin }),
            label,
          ).toEqual(expected);
        }
      },
    );

    it.each(PRIVACY_AFTER_DISCONNECT_ROWS)(
      "disconnect $disconnect, whatever the apps state and origin",
      ({ disconnect, expected }) => {
        for (const { planState, pending, label } of PLAN_AND_PENDING) {
          for (const [appsKey, apps] of Object.entries(APPS)) {
            for (const origin of ORIGINS) {
              expect(
                panel({
                  planState,
                  pending,
                  apps,
                  origin,
                  disconnect: DISCONNECTS[disconnect],
                }),
                `${label}, apps ${appsKey}, from ${origin}`,
              ).toEqual(expected);
            }
          }
        }
      },
    );

    it.each(["done", "failed"] as const)(
      "ignores a %s disconnect that names no apps",
      (status) => {
        for (const { apps, origin } of PRIVACY_IDLE_ROWS) {
          expect(
            panel({
              apps: APPS[apps],
              origin,
              disconnect: { status, names: [] },
            }),
          ).toEqual(panel({ apps: APPS[apps], origin }));
        }
      },
    );
  });

  it("shows nothing until a reason is picked", () => {
    for (const { planState, pending } of PLAN_AND_PENDING) {
      for (const context of CONTEXTS) {
        expect(
          panel({
            reason: null,
            planState,
            pending,
            apps: context.apps,
            origin: context.origin,
            disconnect: context.disconnect,
          }),
          context.label,
        ).toBeNull();
      }
    }
  });

  it.each([
    ["managed", "managed"],
    ["null", null],
  ] as const)(
    "plan state %s only ever shows the privacy panel",
    (_, planState) => {
      for (const reason of NON_PRIVACY_REASONS) {
        for (const pending of Object.values(PENDING)) {
          expect(panel({ reason, planState, pending })).toBeNull();
        }
      }
      expect(panel({ planState, apps: APPS.two })).toEqual(
        offer("Gmail and Google Calendar", MANAGE_APPS),
      );
    },
  );

  it("points max at Pro's cancel panel for not_using unless a cancel is pending", () => {
    expect(panel({ reason: "not_using", planState: "max" })).toEqual(
      PRO_CANCEL,
    );
    expect(
      panel({
        reason: "not_using",
        planState: "max",
        pending: PENDING.downgrade,
      }),
    ).toEqual(PRO_CANCEL);
    expect(
      panel({ reason: "not_using", planState: "max", pending: PENDING.cancel }),
    ).toEqual(PLAN_ENDS);
  });

  it("offers Max for limits only on pro without a pending cancel", () => {
    expect(panel({ reason: "limits", planState: "pro" })).toEqual(LIMITS);
    expect(
      panel({ reason: "limits", planState: "pro", pending: PENDING.downgrade }),
    ).toEqual(LIMITS);
    expect(
      panel({ reason: "limits", planState: "pro", pending: PENDING.cancel }),
    ).toBeNull();
    for (const planState of [
      "max",
      "trial",
      "none",
      "managed",
      null,
    ] as const) {
      expect(panel({ reason: "limits", planState })).toBeNull();
    }
  });

  describe("the Pro price", () => {
    const DOWNGRADE_ONLY =
      "A downgrade keeps your Experts, chats and schedules and starts when your current period ends.";

    it.each([
      [5000, `Pro is $50.00 / month. ${DOWNGRADE_ONLY}`],
      [4999, `Pro is $49.99 / month. ${DOWNGRADE_ONLY}`],
      [null, DOWNGRADE_ONLY],
      [0, DOWNGRADE_ONLY],
    ])("max cost panel with Pro at %s cents", (proMonthlyCents, copy) => {
      expect(
        panel({ reason: "cost", planState: "max", proMonthlyCents }),
      ).toEqual({ copy, action: BILLING });
    });

    it("changes no other panel", () => {
      for (const reason of [...NON_PRIVACY_REASONS, "privacy"] as const) {
        for (const { planState, pending, label } of PLAN_AND_PENDING) {
          if (reason === "cost" && planState === "max" && pending === null) {
            continue;
          }
          const priced = panel({ reason, planState, pending, apps: APPS.one });
          for (const proMonthlyCents of [null, 0, 4999]) {
            expect(
              panel({
                reason,
                planState,
                pending,
                apps: APPS.one,
                proMonthlyCents,
              }),
              `${reason}, ${label}, ${proMonthlyCents}`,
            ).toEqual(priced);
          }
        }
      }
    });
  });

  describe("pending dates", () => {
    it.each([
      ["a Date", ENDS_ON],
      ["epoch milliseconds", ENDS_ON.getTime()],
      ["an ISO string", new Date(2026, 10, 30, 12).toISOString()],
    ])("formats %s through formatShortDate", (_, effectiveAt) => {
      const date = formatShortDate(effectiveAt);
      expect(date).toBe("Nov 30, 2026");

      const cancel: PendingChange = { kind: "cancel", effectiveAt };
      const downgrade: PendingChange = { kind: "downgrade", effectiveAt };
      expect(
        panel({ reason: "cost", planState: "pro", pending: cancel })?.copy,
      ).toBe(
        `Your plan already ends on ${date}, and you won't be charged again. Your account stays after that.`,
      );
      expect(
        panel({ reason: "cost", planState: "trial", pending: cancel })?.copy,
      ).toBe(`Your trial already ends on ${date} and you won't be charged.`);
      expect(
        panel({ reason: "cost", planState: "max", pending: downgrade })?.copy,
      ).toBe(`Your switch to Pro is already scheduled for ${date}.`);
    });

    it("uses the date it is given", () => {
      const effectiveAt = new Date(2027, 0, 5);
      expect(
        panel({
          reason: "cost",
          planState: "max",
          pending: { kind: "cancel", effectiveAt },
        })?.copy,
      ).toBe(
        "Your plan already ends on Jan 5, 2027, and you won't be charged again. Your account stays after that.",
      );
    });
  });

  describe("the trial end date", () => {
    const TRIAL_ENDS_ON = new Date(2026, 9, 20);

    it.each(["cost", "not_using"] as const)(
      "%s: a canceled trial keeps access until it ends",
      (reason) => {
        expect(
          panel({ reason, planState: "trial", trialEndsAt: TRIAL_ENDS_ON }),
        ).toEqual({
          copy: "You can cancel your trial instead. You won't be charged and you keep access until Oct 20, 2026.",
          action: BILLING,
        });
      },
    );

    it.each(["cost", "not_using"] as const)(
      "%s: without an end date the clause is left out",
      (reason) => {
        expect(panel({ reason, planState: "trial" })).toEqual(TRIAL_CANCEL);
      },
    );

    it("a trial that already ends keeps its own panel", () => {
      expect(
        panel({
          reason: "cost",
          planState: "trial",
          pending: PENDING.cancel,
          trialEndsAt: TRIAL_ENDS_ON,
        }),
      ).toEqual(TRIAL_ENDS);
    });

    it.each(["max", "pro", "none", "managed"] as const)(
      "changes nothing for %s",
      (planState) => {
        for (const reason of ["cost", "not_using", "limits"] as const) {
          expect(
            panel({ reason, planState, trialEndsAt: TRIAL_ENDS_ON }),
          ).toEqual(panel({ reason, planState }));
        }
      },
    );
  });
});

describe("formatAppNames", () => {
  it.each([
    [0, ""],
    [1, "Gmail"],
    [2, "Gmail and Google Calendar"],
    [3, "Gmail, Google Calendar and 1 more app"],
    [4, "Gmail, Google Calendar and 2 more apps"],
    [5, "Gmail, Google Calendar and 3 more apps"],
  ])("formats %i names as %j", (count, expected) => {
    expect(formatAppNames(APP_NAMES.slice(0, count))).toBe(expected);
  });
});

describe("getPlanState", () => {
  const TIER_ROWS: [SubscriptionStatusResponseTier, PlanState][] = [
    ["MAX", "max"],
    ["PRO", "pro"],
    ["TRIAL", "trial"],
    ["NO_TIER", "none"],
    ["BASIC", "managed"],
    ["BUSINESS", "managed"],
    ["ENTERPRISE", "managed"],
  ];

  it("covers every tier the API can return", () => {
    expect(TIER_ROWS.map(([tier]) => tier).sort()).toEqual(
      Object.values(SubscriptionStatusResponseTier).sort(),
    );
  });

  it.each(TIER_ROWS)("maps %s to %s", (tier, expected) => {
    expect(getPlanState({ tier, isAdminManaged: false })).toBe(expected);
  });

  it.each(TIER_ROWS)("treats an admin-managed %s as managed", (tier) => {
    expect(getPlanState({ tier, isAdminManaged: true })).toBe("managed");
  });

  it.each([undefined, null])(
    "returns null while the subscription is loading or failed (tier %s)",
    (tier) => {
      expect(getPlanState({ tier, isAdminManaged: false })).toBeNull();
      expect(getPlanState({ tier, isAdminManaged: true })).toBeNull();
    },
  );

  it("returns null for a tier this client doesn't know yet", () => {
    const tier = "FUTURE" as SubscriptionStatusResponseTier;
    expect(getPlanState({ tier, isAdminManaged: false })).toBeNull();
  });
});

describe("EXIT_REASON_OPTIONS", () => {
  it("lists the six reasons in order with their labels", () => {
    expect(EXIT_REASON_OPTIONS).toEqual([
      { key: "privacy", label: "I'm worried about my data or privacy" },
      { key: "cost", label: "It costs too much" },
      { key: "not_using", label: "I'm not using it enough" },
      { key: "missing_capability", label: "It couldn't do what I needed" },
      { key: "limits", label: "I kept hitting usage limits" },
      { key: "other", label: "Something else" },
    ]);
  });
});

describe("getTextPlaceholder", () => {
  it.each([
    ["privacy", "Anything else we should know? (optional)"],
    ["cost", "Anything else we should know? (optional)"],
    ["not_using", "Anything else we should know? (optional)"],
    ["missing_capability", "What were you trying to get done?"],
    ["limits", "Anything else we should know? (optional)"],
    ["other", "Tell us more (optional)"],
    [null, "Anything else we should know? (optional)"],
  ] as const)("uses the %s placeholder", (reason, expected) => {
    expect(getTextPlaceholder(reason)).toBe(expected);
  });
});
