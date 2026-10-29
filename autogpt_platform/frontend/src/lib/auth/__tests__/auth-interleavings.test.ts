import type BackendAPI from "@/lib/autogpt-server-api/client";
import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Randomized interleavings of the auth state machine across several tabs.
 *
 * Everything runs for real (auth store, auth helpers, impersonation storage,
 * header builder, beginImpersonation) except the edges of the browser: the
 * server actions, the audit request, timers and other tabs. Those are driven
 * by a seeded scheduler that completes requests in random order, fails some
 * in transit, fires timers late, reloads tabs and delivers cross-tab storage
 * events. Server answers are computed when a request is made, so an answer
 * that arrives late is as stale as it would be in a browser.
 *
 * After every step, in every tab:
 *  (a) X-Act-As-User-Id is only built for a signed-in admin who owns the
 *      stored target;
 *  (b) once a sign-out is confirmed, no tab is signed in or acting as anyone
 *      until somebody signs in again;
 *  (c) no stored target (cookie or any tab's copy) is paired with an owner
 *      other than the admin who chose it.
 *  (d) at most one session check (start-up lookup, validation or refresh) is in
 *      flight per tab for the current session epoch.
 *  (e) the target a tab sends matches the shared cookie the server checks.
 */

type Kind =
  | "getCurrentUser"
  | "validateSession"
  | "refreshSession"
  | "serverLogout"
  | "audit";

interface ServerUser {
  id: string;
  email: string;
  role: "admin" | "authenticated";
}

const harness = vi.hoisted(() => ({
  request: (_kind: string, _args: unknown[]): Promise<unknown> => {
    throw new Error("harness not ready");
  },
  navigate: (_path: string): void => undefined,
}));

vi.mock("@/lib/auth/actions", () => ({
  getCurrentUser: (...args: unknown[]) =>
    harness.request("getCurrentUser", args),
  validateSession: (...args: unknown[]) =>
    harness.request("validateSession", args),
  refreshSession: (...args: unknown[]) =>
    harness.request("refreshSession", args),
  serverLogout: (...args: unknown[]) => harness.request("serverLogout", args),
}));
vi.mock("@/lib/auth/helpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/helpers")>()),
  // A full page load is the browser's job; the scheduler reloads the tab.
  navigateWithFullLoad: (path: string) => harness.navigate(path),
  reloadPage: () => harness.navigate("reload"),
}));
vi.mock("@/lib/react-query/queryClient", () => ({
  getQueryClient: () => ({ clear() {}, invalidateQueries: async () => {} }),
}));
vi.mock("@/services/analytics/reset-identity", () => ({
  resetAnalyticsIdentity() {},
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast() {},
  useToast: () => ({ toast() {} }),
}));

import { ImpersonationCookie as cookieWriter } from "@/lib/impersonation";

type StoreModule = typeof import("@/lib/auth/hooks/useAuthStore");
type ImpersonationModule = typeof import("@/lib/impersonation");
type BeginModule =
  typeof import("@/app/(platform)/admin/components/beginImpersonation");

interface Tab {
  name: string;
  session: Map<string, string>;
  listeners: Map<string, Set<(event: Event) => void>>;
  store: StoreModule["useAuthStore"];
  imp: ImpersonationModule;
  begin: BeginModule["beginImpersonation"];
  alive: boolean;
  // The record this tab was restored with, inert until its startup check.
  restoredWith?: string;
}

interface Pending {
  tab: Tab;
  kind: Kind;
  // The tab's session epoch when the request was made.
  epoch: number;
  answer: unknown;
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
}

interface Timer {
  id: number;
  tab: Tab;
  run: () => void;
}

const ADMIN_A: ServerUser = {
  id: "admin-a",
  email: "a@x.invalid",
  role: "admin",
};
const ADMIN_B: ServerUser = {
  id: "admin-b",
  email: "b@x.invalid",
  role: "admin",
};
const USER_C: ServerUser = {
  id: "user-c",
  email: "c@x.invalid",
  role: "authenticated",
};
const PEOPLE = [ADMIN_A, ADMIN_B, USER_C];

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const realSetTimeout = globalThis.setTimeout;
const realClearTimeout = globalThis.clearTimeout;
const realWindowAdd = window.addEventListener.bind(window);
const realWindowRemove = window.removeEventListener.bind(window);
const realDocumentAdd = document.addEventListener.bind(document);
const realDocumentRemove = document.removeEventListener.bind(document);

// Lets every continuation a step set off run to completion. Nothing here
// waits on a real clock; a couple of macrotask turns cover dynamic imports.
async function settle() {
  for (let i = 0; i < 2; i++) {
    await new Promise((resolve) => realSetTimeout(resolve, 0));
  }
}

function makeRouter() {
  const noop = () => undefined;
  return {
    push: noop,
    replace: noop,
    refresh: noop,
    back: noop,
    forward: noop,
    prefetch: noop,
  } as unknown as AppRouterInstance;
}

// Cookies are written the way production writes them (Secure,
// SameSite=Lax), through the impersonation module's own cookie writer.
function expireAuthCookies() {
  cookieWriter.clear();
}

class World {
  rand: () => number;
  server: { session: ServerUser | null } = { session: null };
  tabs: Tab[] = [];
  current: Tab | null = null;
  pending: Pending[] = [];
  timers: Timer[] = [];
  timerId = 0;
  owners = new Map<string, string>();
  nextTarget = 0;
  signedOutForGood = false;
  log: string[] = [];
  // Work that ran outside any tab: a continuation escaped the scheduler.
  escaped = 0;

  constructor(seed: number) {
    this.rand = mulberry32(seed);
  }

  pick<T>(items: T[]): T {
    return items[Math.floor(this.rand() * items.length)];
  }

  // ---- tabs ---------------------------------------------------------------

  private enter(tab: Tab) {
    sessionStorage.clear();
    for (const [key, value] of tab.session) sessionStorage.setItem(key, value);
    this.current = tab;
  }

  private leave(tab: Tab) {
    tab.session = new Map(
      Object.keys(sessionStorage).map((key) => [
        key,
        sessionStorage.getItem(key) ?? "",
      ]),
    );
    this.current = null;
  }

  /** Runs `work` as `tab`, lets everything it set off finish, then delivers
   * any cross-tab storage events it caused. */
  async inTab(tab: Tab, work: () => unknown) {
    if (!tab.alive) return;
    const before = new Map(
      Object.keys(localStorage).map((key) => [key, localStorage.getItem(key)]),
    );
    this.enter(tab);
    try {
      const result = work();
      if (result instanceof Promise) result.catch(() => undefined);
      await settle();
    } finally {
      this.leave(tab);
    }
    for (const key of Object.keys(localStorage)) {
      const value = localStorage.getItem(key);
      if (before.get(key) === value) continue;
      for (const other of this.tabs) {
        if (other === tab || !other.alive) continue;
        const handlers = [...(other.listeners.get("storage") ?? [])];
        await this.inTab(other, () => {
          for (const handler of handlers) {
            handler(new StorageEvent("storage", { key, newValue: value }));
          }
        });
      }
    }
  }

  async openTab(name: string, session = new Map<string, string>()) {
    vi.resetModules();
    const [{ useAuthStore }, imp, { beginImpersonation }] = await Promise.all([
      import("@/lib/auth/hooks/useAuthStore"),
      import("@/lib/impersonation"),
      import("@/app/(platform)/admin/components/beginImpersonation"),
    ]);
    const tab: Tab = {
      name,
      session,
      listeners: new Map(),
      store: useAuthStore,
      imp,
      begin: beginImpersonation,
      alive: true,
    };
    await this.inTab(tab, () =>
      useAuthStore.getState().initialize({
        api: { disconnectWebSocket() {} } as unknown as BackendAPI,
        router: makeRouter(),
        path: "/library",
      }),
    );
    return tab;
  }

  async reload(tab: Tab) {
    tab.alive = false;
    this.pending = this.pending.filter((p) => p.tab !== tab);
    this.timers = this.timers.filter((t) => t.tab !== tab);
    const fresh = await this.openTab(tab.name, tab.session);
    fresh.restoredWith = tab.restoredWith;
    this.tabs[this.tabs.indexOf(tab)] = fresh;
  }

  // ---- the edges ------------------------------------------------------------

  request(kind: Kind): Promise<unknown> {
    const tab = this.current;
    if (!tab) throw new Error(`${kind} requested outside a tab`);
    const user = this.server.session ? { ...this.server.session } : null;
    let answer: unknown;
    if (kind === "getCurrentUser") answer = { user };
    else if (kind === "validateSession")
      answer = user
        ? { user, isValid: true }
        : {
            user: null,
            isValid: false,
            redirectPath: "/login?next=%2Flibrary",
          };
    else if (kind === "refreshSession")
      answer = user ? { user } : { user: null, error: "No active session" };
    else answer = { success: true };
    const epoch = tab.store.getState().sessionEpoch;
    return new Promise((resolve, reject) => {
      this.pending.push({ tab, kind, epoch, answer, resolve, reject });
    });
  }

  install() {
    harness.request = (kind) => this.request(kind as Kind);
    harness.navigate = () => {
      const tab = this.current;
      if (tab) this.reloadRequests.push(tab);
    };
    globalThis.setTimeout = ((run: () => void) => {
      const tab = this.current;
      const id = ++this.timerId;
      if (tab) this.timers.push({ id, tab, run });
      else this.escaped += 1;
      return id;
    }) as unknown as typeof setTimeout;
    globalThis.clearTimeout = ((id: number) => {
      this.timers = this.timers.filter((t) => t.id !== id);
    }) as unknown as typeof clearTimeout;
    const capture =
      (real: typeof window.addEventListener) =>
      (
        type: string,
        handler: EventListenerOrEventListenerObject,
        opts?: unknown,
      ) => {
        if (
          this.current &&
          ["storage", "online", "visibilitychange"].includes(type)
        ) {
          const set = this.current.listeners.get(type) ?? new Set();
          set.add(handler as (event: Event) => void);
          this.current.listeners.set(type, set);
          return;
        }
        real(type, handler, opts as AddEventListenerOptions);
      };
    const release =
      (real: typeof window.removeEventListener) =>
      (
        type: string,
        handler: EventListenerOrEventListenerObject,
        opts?: unknown,
      ) => {
        for (const tab of this.tabs) {
          tab.listeners.get(type)?.delete(handler as (event: Event) => void);
        }
        real(type, handler, opts as EventListenerOptions);
      };
    window.addEventListener = capture(
      realWindowAdd,
    ) as typeof window.addEventListener;
    window.removeEventListener = release(
      realWindowRemove,
    ) as typeof window.removeEventListener;
    document.addEventListener = capture(
      realDocumentAdd,
    ) as typeof document.addEventListener;
    document.removeEventListener = release(
      realDocumentRemove,
    ) as typeof document.removeEventListener;
  }

  reloadRequests: Tab[] = [];

  // ---- the user and the server ------------------------------------------------

  async step() {
    const tab = this.pick(this.tabs);
    const roll = this.rand();
    if (roll < 0.1) {
      const person = this.pick(PEOPLE);
      this.log.push(`${tab.name}: sign in as ${person.id}`);
      // What the login page does before the session exists.
      await this.inTab(tab, () => tab.imp.clearImpersonationEverywhere());
      this.server.session = { ...person };
      this.signedOutForGood = false;
      // Signing in lands back in the app, which initializes the store.
      await this.inTab(tab, () =>
        tab.store.getState().initialize({
          api: { disconnectWebSocket() {} } as unknown as BackendAPI,
          router: makeRouter(),
          path: "/library",
        }),
      );
    } else if (roll < 0.16) {
      const session = this.server.session;
      if (session) {
        session.role = session.role === "admin" ? "authenticated" : "admin";
        this.log.push(`server: ${session.id} is now ${session.role}`);
      }
    } else if (roll < 0.2) {
      this.log.push("server: session expires");
      this.server.session = null;
    } else if (roll < 0.32) {
      this.log.push(`${tab.name}: validate`);
      await this.inTab(tab, () =>
        tab.store.getState().validateSession({ force: this.rand() < 0.5 }),
      );
    } else if (roll < 0.4) {
      this.log.push(`${tab.name}: refresh`);
      await this.inTab(tab, () => tab.store.getState().refreshSession());
    } else if (roll < 0.43) {
      this.log.push(`${tab.name}: back online`);
      const handlers = [...(tab.listeners.get("online") ?? [])];
      await this.inTab(tab, () =>
        handlers.forEach((h) => h(new Event("online"))),
      );
    } else if (roll < 0.46) {
      await this.focus(tab);
    } else if (roll < 0.54) {
      this.log.push(`${tab.name}: sign out`);
      await this.inTab(tab, () => tab.store.getState().logOut());
    } else if (roll < 0.64) {
      const target = `target-${++this.nextTarget}`;
      const owner = tab.store.getState().user?.id;
      if (owner) this.owners.set(target, owner);
      this.log.push(
        `${tab.name}: impersonate ${target} as ${owner ?? "nobody"}`,
      );
      await this.inTab(tab, () =>
        tab.begin(
          target,
          () =>
            new Promise((resolve, reject) => {
              this.pending.push({
                tab,
                kind: "audit",
                epoch: tab.store.getState().sessionEpoch,
                answer: {},
                resolve,
                reject,
              });
            }),
        ),
      );
    } else if (roll < 0.67) {
      this.log.push(`${tab.name}: reload`);
      await this.reload(tab);
    } else if (roll < 0.7) {
      // A browser restoring a tab brings back its sessionStorage, which may
      // hold an old target (another admin's, or a bare id from before owners
      // were recorded) after the cleanup signal went out without it.
      // Prefer another admin's old target than the one in the shared cookie
      // now: the mix a tab must never blend into one record.
      const cookieOwner = this.peekCookieOwner();
      const known = [...this.owners.entries()];
      const foreign = known.filter(([, owner]) => owner !== cookieOwner);
      const pool = foreign.length ? foreign : known;
      const stale = pool.length && this.rand() < 0.8 ? this.pick(pool) : null;
      const raw = stale
        ? JSON.stringify({ target: stale[0], owner: stale[1] })
        : "bare-target-id";
      this.log.push(`${tab.name}: restored with ${raw}`);
      tab.session = new Map([["admin-impersonate-user-id", raw]]);
      await this.reload(tab);
      const restored = this.tabs.find((t) => t.name === tab.name);
      if (restored?.alive) restored.restoredWith = raw;
    } else if (roll < 0.73 && !this.signedOutForGood && this.owners.size) {
      // A shared browser: another admin's record left in the cookie by a
      // session that ended without a clean sign-out.
      const [target, owner] = this.pick([...this.owners.entries()]);
      this.log.push(`cookie left behind: ${target} for ${owner}`);
      cookieWriter.set({ target, owner });
    } else if (roll < 0.94 && this.pending.length) {
      const op = this.pick(this.pending);
      this.pending.splice(this.pending.indexOf(op), 1);
      const lost = this.rand() < 0.2;
      this.log.push(
        `${op.tab.name}: ${op.kind} ${lost ? "lost in transit" : "answered"}`,
      );
      if (!lost && op.kind === "serverLogout") {
        // The server ends the session and expires its cookies on delivery.
        this.server.session = null;
        expireAuthCookies();
        this.signedOutForGood = true;
      }
      await this.inTab(op.tab, () =>
        lost
          ? op.reject(new TypeError("Failed to fetch"))
          : op.resolve(op.answer),
      );
    } else if (this.timers.length) {
      const timer = this.pick(this.timers);
      this.timers.splice(this.timers.indexOf(timer), 1);
      this.log.push(`${timer.tab.name}: timer fires`);
      await this.inTab(timer.tab, () => timer.run());
    }

    while (this.reloadRequests.length) {
      const reloading = this.reloadRequests.shift()!;
      if (reloading.alive) await this.reload(reloading);
    }
  }

  /**
   * The tab comes back into view. If a retry was waiting (and no check is in
   * flight), the user shouldn't wait it out: that timer must be gone and
   * exactly one check must go out now, so no stale timer fires a duplicate.
   */
  async focus(tab: Tab) {
    this.log.push(`${tab.name}: focus`);
    const before = tab.store.getState() as {
      retryTimer?: unknown;
      isValidating?: boolean;
    };
    const waitingRetry =
      before.retryTimer != null && !before.isValidating
        ? before.retryTimer
        : null;
    const checksBefore = this.pendingChecks(tab);

    const handlers = [...(tab.listeners.get("visibilitychange") ?? [])];
    await this.inTab(tab, () =>
      handlers.forEach((h) => h(new Event("visibilitychange"))),
    );

    if (waitingRetry != null) {
      expect(
        this.timers.some((t) => t.id === waitingRetry),
        `${tab.name} kept waiting on its old retry after coming back`,
      ).toBe(false);
      expect(
        this.pendingChecks(tab) - checksBefore,
        `${tab.name} should check exactly once on coming back`,
      ).toBe(1);
    }
  }

  pendingChecks(tab: Tab): number {
    return this.pending.filter(
      (p) => p.tab === tab && p.kind === "validateSession",
    ).length;
  }

  peekCookieOwner(): string | null {
    const raw = document.cookie
      .split("; ")
      .find((row) => row.startsWith("admin-impersonate-user-id="))
      ?.split("=")[1];
    try {
      return raw ? (JSON.parse(decodeURIComponent(raw)).owner ?? null) : null;
    } catch {
      return null;
    }
  }

  // ---- invariants ---------------------------------------------------------------

  check() {
    expect(this.escaped, "work escaped the scheduler").toBe(0);
    for (const tab of this.tabs) {
      this.enter(tab);
      try {
        const user = tab.store.getState().user;

        // (d) one session check at a time: start-up's lookups and every
        // validation share it. Requests from an earlier epoch don't count;
        // their answers are dropped.
        const epoch = tab.store.getState().sessionEpoch;
        const checksInFlight = this.pending.filter(
          (p) =>
            p.tab === tab &&
            p.epoch === epoch &&
            (p.kind === "getCurrentUser" ||
              p.kind === "validateSession" ||
              p.kind === "refreshSession"),
        ).length;
        expect(
          checksInFlight,
          `${tab.name} has ${checksInFlight} session checks in flight`,
        ).toBeLessThanOrEqual(1);
        const headers = tab.imp.getSystemHeaders();
        const actingAs = headers[IMPERSONATION_HEADER_NAME];
        const record = tab.imp.ImpersonationState.read();

        if (actingAs) {
          // (a)
          expect(
            user,
            `${tab.name} acts as ${actingAs} while signed out`,
          ).not.toBeNull();
          expect(
            user?.role,
            `${tab.name} acts as ${actingAs} as a non-admin`,
          ).toBe("admin");
          expect(
            record?.owner,
            `${tab.name} acts as another admin's target`,
          ).toBe(user?.id);
          expect(record?.target).toBe(actingAs);
        }

        if (this.signedOutForGood) {
          // (b)
          expect(
            user,
            `${tab.name} signed back in after a confirmed sign-out`,
          ).toBeNull();
          expect(
            actingAs,
            `${tab.name} acts as someone after sign-out`,
          ).toBeUndefined();
          // Only what a restored tab brought back may still sit there.
          if (record && JSON.stringify(record) !== tab.restoredWith) {
            expect.fail(`${tab.name} holds ${record.target} after sign-out`);
          }
          expect(
            tab.imp.ImpersonationCookie.get(),
            "the shared cookie holds a target after sign-out",
          ).toBeNull();
        }

        // (e) the target this tab sends is the one the server will check it
        // against (the shared cookie), even after another tab chose another.
        if (actingAs) {
          expect(
            tab.imp.ImpersonationCookie.get()?.target,
            `${tab.name} sends ${actingAs} but the cookie says otherwise`,
          ).toBe(actingAs);
        }

        // (c)
        for (const stored of [record, tab.imp.ImpersonationCookie.get()]) {
          if (stored) {
            expect(
              this.owners.get(stored.target),
              `${stored.target} paired with ${stored.owner}`,
            ).toBe(stored.owner);
          }
        }
      } finally {
        this.leave(tab);
      }
    }
  }
}

const SEEDS = Number(process.env.AUTH_INTERLEAVING_SEEDS ?? 150);
const STEPS = 40;

describe("auth state machine under random interleavings", () => {
  let world: World | null = null;

  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    expireAuthCookies();
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => "visible",
    });
  });

  afterEach(() => {
    globalThis.setTimeout = realSetTimeout;
    globalThis.clearTimeout = realClearTimeout;
    window.addEventListener = realWindowAdd;
    window.removeEventListener = realWindowRemove;
    document.addEventListener = realDocumentAdd;
    document.removeEventListener = realDocumentRemove;
    world = null;
  });

  it(`holds its invariants across ${SEEDS} seeded runs`, async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    vi.spyOn(console, "debug").mockImplementation(() => undefined);

    const firstSeed = Number(process.env.AUTH_INTERLEAVING_FROM ?? 1);
    for (let seed = firstSeed; seed < firstSeed + SEEDS; seed++) {
      localStorage.clear();
      sessionStorage.clear();
      expireAuthCookies();
      world = new World(seed);
      world.install();
      world.server.session = { ...world.pick(PEOPLE) };
      world.tabs = [await world.openTab("tab-1"), await world.openTab("tab-2")];
      if (world.rand() < 0.5) world.tabs.push(await world.openTab("tab-3"));

      try {
        for (let i = 0; i < STEPS; i++) {
          await world.step();
          world.check();
        }
      } catch (error) {
        throw new Error(
          `seed ${seed} broke an invariant after:\n  ${world.log.join("\n  ")}\n${String(error)}`,
        );
      } finally {
        for (const tab of world.tabs) tab.alive = false;
      }
    }
  }, 120_000);
});
