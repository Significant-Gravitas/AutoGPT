import { describe, expect, it, vi } from "vitest";
import {
  AUTH_EMAILS_PER_IP,
  type AuthEmailContext,
  claimEmailSlot,
  claimIPEmailSlot,
} from "../auth-email-cooldown";
import { sendAuthEmail } from "../email";
import { capAuthEmailsPerIP } from "../ip-email-cap";
import { sendVerificationLink } from "../verification-link";

vi.mock("../email", () => ({ sendAuthEmail: vi.fn() }));

interface Row {
  id: string;
  identifier: string;
  expiresAt: Date;
}

// The verification table as Postgres keeps it: the id is the primary key. Every
// check for a live row waits until all of the burst has made one, which is the
// worst interleaving of sends that arrive together.
function verificationTable(burst: number) {
  const rows: Row[] = [];
  let checked = 0;
  let releaseChecks = () => {};
  const allChecked = new Promise<void>((resolve) => {
    releaseChecks = resolve;
  });
  let n = 0;
  const context: AuthEmailContext = {
    baseURL: "http://localhost:3000/api/auth",
    password: { hash: async (password) => password },
    adapter: {
      findMany: async ({ where }) => {
        const [match] = where;
        const found = rows.filter((row) =>
          match.field === "id"
            ? row.id === match.value
            : row.identifier === match.value && row.expiresAt > new Date(),
        );
        if (++checked === burst) releaseChecks();
        if (checked <= burst) await allChecked;
        return found;
      },
      create: async ({ data, forceAllowId }) => {
        const id = forceAllowId ? String(data.id) : `random-${++n}`;
        if (rows.some((row) => row.id === id)) {
          throw new Error(`duplicate key value violates unique constraint`);
        }
        rows.push({
          id,
          identifier: String(data.identifier),
          expiresAt: data.expiresAt as Date,
        });
        return data;
      },
      updateMany: async () => 0,
      deleteMany: async () => 0,
    },
  };
  return { context, rows };
}

describe("claimEmailSlot", () => {
  it("lets one of a burst of simultaneous sends to an address through", async () => {
    const { context } = verificationTable(10);

    const claims = await Promise.all(
      Array.from({ length: 10 }, () =>
        claimEmailSlot(context, "verify-email", "burst@example.com"),
      ),
    );

    expect(claims.filter(Boolean)).toHaveLength(1);
  });

  it("keeps separate slots per address and per kind of email", async () => {
    const { context } = verificationTable(0);

    expect(await claimEmailSlot(context, "verify-email", "a@example.com")).toBe(
      true,
    );
    expect(await claimEmailSlot(context, "verify-email", "A@example.com")).toBe(
      false,
    );
    expect(await claimEmailSlot(context, "verify-email", "b@example.com")).toBe(
      true,
    );
    expect(
      await claimEmailSlot(context, "repeat-sign-up", "a@example.com"),
    ).toBe(true);
  });

  it("passes on a failure that is not a lost race", async () => {
    const { context } = verificationTable(0);
    context.adapter.create = async () => {
      throw new Error("database is down");
    };

    await expect(
      claimEmailSlot(context, "verify-email", "down@example.com"),
    ).rejects.toThrow("database is down");
  });
});

describe("claimIPEmailSlot", () => {
  it("lets a burst from one IP take only its share of slots", async () => {
    const { context } = verificationTable(0);

    const claims = await Promise.all(
      Array.from({ length: 20 }, () =>
        claimIPEmailSlot(context, "203.0.113.7"),
      ),
    );

    expect(claims.filter(Boolean)).toHaveLength(AUTH_EMAILS_PER_IP);
    expect(await claimIPEmailSlot(context, "198.51.100.9")).toBe(true);
  });

  it.each([
    ["at the start of a window", 0],
    ["midway", 300_000],
    ["just before it ends", 599_999],
  ])(
    "keeps a slot until its window ends, claimed %s",
    async (_, intoWindow) => {
      const windowStart = Date.UTC(2026, 9, 6, 17, 10);
      const windowEnd = windowStart + 10 * 60 * 1000;
      vi.useFakeTimers({ now: windowStart + intoWindow });
      try {
        const { context, rows } = verificationTable(0);

        await claimIPEmailSlot(context, "203.0.113.7");

        const expiresAt = rows[0].expiresAt.getTime();
        expect(expiresAt).toBeGreaterThan(Date.now());
        expect(Math.abs(expiresAt - windowEnd)).toBeLessThan(1);
      } finally {
        vi.useRealTimers();
      }
    },
  );

  it("passes on a failure that is not a taken slot", async () => {
    const { context } = verificationTable(0);
    context.adapter.create = async () => {
      throw new Error("database is down");
    };

    await expect(claimIPEmailSlot(context, "203.0.113.7")).rejects.toThrow(
      "database is down",
    );
  });
});

describe("when a cap can't be checked", () => {
  function brokenContext() {
    const { context } = verificationTable(0);
    const down = async () => {
      throw new Error("database is down");
    };
    context.adapter.findMany = down;
    context.adapter.create = down;
    return context;
  }

  it("still sends the verification email", async () => {
    vi.mocked(sendAuthEmail).mockClear();

    await sendVerificationLink({
      user: { id: "user-1", email: "down@example.com" },
      url: "http://localhost:3000/api/auth/verify-email?token=t",
      getAuthContext: async () => brokenContext(),
      hasPlatformUser: async () => false,
      requireEmailVerification: true,
      resetRedirectTo: "http://localhost:3000/reset-password",
    });

    expect(sendAuthEmail).toHaveBeenCalledWith(
      expect.objectContaining({ to: "down@example.com", type: "verify_email" }),
    );
  });

  it("lets the sign-up through", async () => {
    await expect(
      capAuthEmailsPerIP({
        path: "/sign-up/email",
        headers: new Headers({ "x-forwarded-for": "203.0.113.7" }),
        context: { ...brokenContext(), options: {} },
      }),
    ).resolves.toBeUndefined();
  });
});
