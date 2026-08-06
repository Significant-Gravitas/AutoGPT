import { AUTH_PASSWORD_MIN_LENGTH } from "../src/lib/auth/password-policy";
import {
  type QueryExecutor,
  seedRoster as seedRosterDefault,
} from "./seed-preview-accounts.helpers";

export const SEEDED_EXIT_CODE = 0;
export const FAILURE_EXIT_CODE = 1;
export const NO_BETTER_AUTH_EXIT_CODE = 3;

interface TableProbe {
  query(
    text: string,
    params?: unknown[],
  ): Promise<{ rows: { reg?: string | null }[] }>;
}

interface SeedClient extends QueryExecutor {
  release(): void;
}

interface SeedPool extends TableProbe {
  connect(): Promise<SeedClient>;
}

interface SeedOptions {
  schema: string;
  identityTable: string;
  accountTable: string;
  password: string;
}

interface SeedDependencies {
  hashPassword(password: string): Promise<string>;
  seedRoster?: typeof seedRosterDefault;
  reportLog?: (message: string) => void;
  reportError?: (message: string) => void;
}

export function resolvePreviewConnectionString(
  directURL: string | undefined,
  databaseURL: string | undefined,
  reportWarning: (message: string) => void = console.warn,
) {
  if (directURL) return directURL;
  if (!databaseURL) {
    throw new Error("DIRECT_URL or DATABASE_URL must be set");
  }
  reportWarning(
    "DIRECT_URL is unset; using DATABASE_URL. Ensure the endpoint supports " +
      "an explicit transaction on one checked-out client.",
  );
  return databaseURL;
}

export function validatePreviewPassword(password: string | undefined) {
  if (!password) {
    throw new Error("PREVIEW_ACCOUNTS_PASSWORD must be set");
  }
  if (password.length < AUTH_PASSWORD_MIN_LENGTH) {
    throw new Error(
      `PREVIEW_ACCOUNTS_PASSWORD must be at least ${AUTH_PASSWORD_MIN_LENGTH} characters`,
    );
  }
  return password;
}

export async function probeBetterAuthTables(
  pool: TableProbe,
  identityTable: string,
  accountTable: string,
  schema: string,
) {
  async function tableExists(qualified: string) {
    const { rows } = await pool.query("SELECT to_regclass($1)::text AS reg", [
      qualified,
    ]);
    return Boolean(rows[0]?.reg);
  }

  const hasIdentity = await tableExists(identityTable);
  const hasAccount = await tableExists(accountTable);
  if (!hasIdentity && !hasAccount) return "absent" as const;
  if (!hasIdentity || !hasAccount) {
    throw new Error(
      `Schema "${schema}" has only one of UserAuthIdentity/UserAuthAccount ` +
        "— half-applied migration; refusing to seed or fall back.",
    );
  }
  return "present" as const;
}

export async function rollbackTransaction(
  client: QueryExecutor,
  reportError: (message: string) => void = console.error,
) {
  try {
    await client.query("ROLLBACK");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    reportError(`Preview account seeder rollback failed: ${message}`);
  }
}

export async function runInTransaction<T>(
  client: QueryExecutor,
  operation: () => Promise<T>,
  reportError: (message: string) => void = console.error,
) {
  await client.query("BEGIN");
  try {
    const result = await operation();
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await rollbackTransaction(client, reportError);
    throw error;
  }
}

export async function seedPreviewAccounts(
  pool: SeedPool,
  options: SeedOptions,
  dependencies: SeedDependencies,
) {
  const { schema, identityTable, accountTable, password } = options;
  const reportLog = dependencies.reportLog ?? console.log;
  const reportError = dependencies.reportError ?? console.error;
  const tableState = await probeBetterAuthTables(
    pool,
    identityTable,
    accountTable,
    schema,
  );
  if (tableState === "absent") {
    reportError(
      `Better Auth tables not present in schema "${schema}" — ` +
        "pre-migration database, nothing to seed here.",
    );
    return NO_BETTER_AUTH_EXIT_CODE;
  }

  const passwordHash = await dependencies.hashPassword(password);
  const client = await pool.connect();
  try {
    const seed = dependencies.seedRoster ?? seedRosterDefault;
    const { createdIdentities, createdAccounts } = await runInTransaction(
      client,
      () => seed(client, { identityTable, accountTable, passwordHash }),
      reportError,
    );
    reportLog(
      `Seeded preview accounts: ${createdIdentities} identities and ` +
        `${createdAccounts} credential accounts created.`,
    );
    return SEEDED_EXIT_CODE;
  } finally {
    client.release();
  }
}

export function reportPreviewSeedFailure(
  error: unknown,
  reportError: (message: string) => void = console.error,
) {
  const code =
    error instanceof Error && "code" in error
      ? ` (code ${(error as { code?: string }).code})`
      : "";
  const message = error instanceof Error ? error.message : String(error);
  reportError(`Preview account seeding failed: ${message}${code}`);
  return FAILURE_EXIT_CODE;
}
