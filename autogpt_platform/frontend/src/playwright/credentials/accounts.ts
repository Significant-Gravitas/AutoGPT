import path from "path";

export const SEEDED_TEST_PASSWORD =
  process.env.SEEDED_TEST_PASSWORD || "testpassword123";
export const SEEDED_USER_POOL_VERSION = "2.0.0";

export const SEEDED_TEST_ACCOUNTS = {
  primary: {
    key: "primary",
    email: "test123@example.com",
    password: SEEDED_TEST_PASSWORD,
  },
  smokeAuth: {
    key: "smokeAuth",
    email: "e2e.qa.auth@example.com",
    password: SEEDED_TEST_PASSWORD,
  },
  smokeBuilder: {
    key: "smokeBuilder",
    email: "e2e.qa.builder@example.com",
    password: SEEDED_TEST_PASSWORD,
  },
  smokeLibrary: {
    key: "smokeLibrary",
    email: "e2e.qa.library@example.com",
    password: SEEDED_TEST_PASSWORD,
  },
  smokeMarketplace: {
    key: "smokeMarketplace",
    email: "e2e.qa.marketplace@example.com",
    password: SEEDED_TEST_PASSWORD,
  },
  smokeSettings: {
    key: "smokeSettings",
    email: "e2e.qa.settings@example.com",
    password: SEEDED_TEST_PASSWORD,
  },
  parallelA: {
    key: "parallelA",
    email: "e2e.qa.parallel.a@example.com",
    password: SEEDED_TEST_PASSWORD,
  },
  parallelB: {
    key: "parallelB",
    email: "e2e.qa.parallel.b@example.com",
    password: SEEDED_TEST_PASSWORD,
  },
} as const;

export type SeededTestAccountKey = keyof typeof SEEDED_TEST_ACCOUNTS;
export type SeededTestAccount =
  (typeof SEEDED_TEST_ACCOUNTS)[SeededTestAccountKey];

export const SEEDED_TEST_USERS = Object.values(SEEDED_TEST_ACCOUNTS);
export const SEEDED_AUTH_STATE_ACCOUNT_KEYS = [
  "smokeBuilder",
  "smokeLibrary",
  "smokeMarketplace",
  "smokeSettings",
  "parallelA",
  "parallelB",
] as const;

export const AUTH_DIRECTORY = path.resolve(process.cwd(), ".auth");

export function getAuthStatePath(accountKey: SeededTestAccountKey) {
  return path.join(AUTH_DIRECTORY, "states", `${accountKey}.json`);
}

export const E2E_AUTH_STATES = {
  builder: getAuthStatePath("smokeBuilder"),
  library: getAuthStatePath("smokeLibrary"),
  marketplace: getAuthStatePath("smokeMarketplace"),
  settings: getAuthStatePath("smokeSettings"),
  parallelA: getAuthStatePath("parallelA"),
  parallelB: getAuthStatePath("parallelB"),
} as const;

export const SMOKE_AUTH_STATES = E2E_AUTH_STATES;

export function getSeededTestUser(
  accountKey: SeededTestAccountKey = "primary",
): SeededTestAccount {
  return SEEDED_TEST_ACCOUNTS[accountKey];
}
