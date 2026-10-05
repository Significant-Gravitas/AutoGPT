import { describe, expect, it, vi } from "vitest";
import { type AuthEmailContext, claimEmailSlot } from "../auth-email-cooldown";

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
        const [identifier] = where;
        const found = rows.filter(
          (row) =>
            row.identifier === identifier.value && row.expiresAt > new Date(),
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
