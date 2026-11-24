import { planGate, readPlanCheck, shouldRetryPlanCheck } from "../gate";

describe("readPlanCheck", () => {
  it("blocks an account with no plan on a server that sells plans", () => {
    expect(readPlanCheck(true, "NO_TIER")).toEqual({
      isPaymentEnabled: true,
      hasPlan: false,
    });
  });

  it.each(["TRIAL", "BASIC", "PRO", "MAX", "BUSINESS", "ENTERPRISE"] as const)(
    "lets a %s account in",
    (tier) => {
      expect(readPlanCheck(true, tier).hasPlan).toBe(true);
    },
  );

  it("lets everyone in on a server that sells no plans", () => {
    expect(readPlanCheck(false, null)).toEqual({
      isPaymentEnabled: false,
      hasPlan: true,
    });
  });
});

describe("planGate", () => {
  const blocked = { isPaymentEnabled: true, hasPlan: false };
  const paid = { isPaymentEnabled: true, hasPlan: true };

  it("waits on the launch screen until the server answers", () => {
    expect(planGate({ check: undefined, status: "pending" })).toBe("loading");
  });

  it("follows the answer", () => {
    expect(planGate({ check: paid, status: "success" })).toBe("open");
    expect(planGate({ check: blocked, status: "success" })).toBe("blocked");
  });

  it("lets the person in when the server never answers", () => {
    expect(planGate({ check: undefined, status: "error" })).toBe("open");
  });

  it("keeps an account blocked when a later check fails", () => {
    expect(planGate({ check: blocked, status: "error" })).toBe("blocked");
  });
});

describe("shouldRetryPlanCheck", () => {
  it("asks again while a new plan is still being set up", () => {
    expect(shouldRetryPlanCheck(0, { status: 503 })).toBe(true);
    expect(shouldRetryPlanCheck(3, { status: 503 })).toBe(false);
  });

  it("gives up at once on anything else", () => {
    expect(shouldRetryPlanCheck(0, { status: 500 })).toBe(false);
    expect(shouldRetryPlanCheck(0, new Error("Timed out"))).toBe(false);
  });
});
