import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { PoolConfig } from "pg";
import { afterEach, describe, expect, it, vi } from "vitest";

// The single-container appliance's connection limit for the frontend role.
const ROLE_LIMIT = Number(
  /CONNECTION LIMIT (\d+)/.exec(
    readFileSync(resolve("../single-container/bootstrap.sh"), "utf8"),
  )?.[1],
);

// Postgres as the frontend role sees it: it refuses a connection past the
// role's limit, and counts a closed one until its backend has exited.
const postgres = vi.hoisted(() => ({
  roleLimit: 0,
  open: 0,
  peak: 0,
  random: Math.random,
  yieldTicks(ticks: number) {
    let done = Promise.resolve();
    for (let i = 0; i < ticks; i++) {
      done = done.then(() => new Promise<void>((r) => setImmediate(r)));
    }
    return done;
  },
}));

vi.mock("pg", async (importOriginal) => {
  const pg = await importOriginal<typeof import("pg")>();
  const { EventEmitter } = await import("node:events");

  class FakeClient extends EventEmitter {
    _queryable = true;
    _ending = false;
    private connected = false;

    connect(callback: (error?: Error) => void) {
      setImmediate(() => {
        if (postgres.open >= postgres.roleLimit) {
          const refused = new Error(
            'too many connections for role "autogpt_frontend"',
          );
          callback(Object.assign(refused, { code: "53300" }));
          return;
        }
        this.connected = true;
        postgres.open += 1;
        postgres.peak = Math.max(postgres.peak, postgres.open);
        callback();
      });
    }

    query(
      _text: string,
      _values: unknown,
      callback: (error: Error | null, result?: unknown) => void,
    ) {
      void postgres.yieldTicks(Math.floor(postgres.random() * 4)).then(() =>
        // A failed statement makes the pool throw its connection away.
        postgres.random() < 0.1
          ? callback(new Error("canceling statement due to statement timeout"))
          : callback(null, { rows: [] }),
      );
    }

    end(callback?: () => void) {
      if (!this._ending) {
        this._ending = true;
        this._queryable = false;
        if (this.connected) {
          void postgres.yieldTicks(1).then(() => (postgres.open -= 1));
        }
      }
      callback?.();
      this.emit("end");
    }
  }

  return {
    ...pg,
    Pool: class extends pg.Pool {
      constructor(config?: PoolConfig) {
        super({ ...config, Client: FakeClient as never });
      }
    },
  };
});

// Next bundles route handlers and server actions separately, and each bundle
// evaluates its own copy of the module; a query suffix gives one here.
async function importCopy(name: string) {
  const specifier = `../auth-db-pool?copy=${name}`;
  return (await import(
    /* @vite-ignore */ specifier
  )) as typeof import("../auth-db-pool");
}

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

afterEach(() => {
  vi.unstubAllEnvs();
  delete (globalThis as { __authDbPool?: unknown }).__authDbPool;
});

describe("auth database pool", () => {
  it("keeps one process inside the frontend role's connection limit, whatever the bundling and interleaving", async () => {
    vi.stubEnv("NODE_ENV", "production");
    postgres.roleLimit = ROLE_LIMIT;

    for (let seed = 1; seed <= 300; seed++) {
      const random = seededRandom(seed);
      postgres.random = random;
      postgres.open = 0;
      postgres.peak = 0;
      delete (globalThis as { __authDbPool?: unknown }).__authDbPool;

      const copies = await Promise.all(
        Array.from({ length: 1 + Math.floor(random() * 3) }, (_, i) =>
          importCopy(`${seed}-${i}`),
        ),
      );
      const refusals: unknown[] = [];
      await Promise.all(
        Array.from({ length: 10 + Math.floor(random() * 60) }, async () => {
          const { authDbPool } = copies[Math.floor(random() * copies.length)];
          await postgres.yieldTicks(Math.floor(random() * 3));
          await authDbPool.query("SELECT 1").catch((error: Error) => {
            if (!error.message.includes("statement timeout")) {
              refusals.push(error.message);
            }
          });
        }),
      );
      await Promise.all(
        [...new Set(copies.map((copy) => copy.authDbPool))].map((pool) =>
          pool.end(),
        ),
      );
      // A closed connection frees its slot a tick later; let those land so
      // they aren't counted against the next seed.
      await postgres.yieldTicks(2);

      expect({ seed, refusals }).toEqual({ seed, refusals: [] });
      expect(postgres.peak).toBeGreaterThan(0);
      expect(postgres.peak).toBeLessThanOrEqual(ROLE_LIMIT);
      expect(postgres.open).toBe(0);
    }
    // 300 seeds take about 4 s alone, more on a busy CI runner.
  }, 30_000);
});
