import { createHash } from "node:crypto";

/**
 * The login accounts every preview environment is seeded with. The
 * "@previews.agpt.co" subdomain dodges the frontend's "@agpt.co" SSO-only
 * login block, and `role` is what the backend's admin gate reads (Better
 * Auth mints it into the JWT role claim).
 */
export const PREVIEW_ACCOUNTS = [
  {
    email: "preview-admin@previews.agpt.co",
    name: "preview-admin",
    role: "admin",
  },
  {
    email: "preview-existing@previews.agpt.co",
    name: "preview-existing",
    role: "user",
  },
  {
    email: "preview-clean@previews.agpt.co",
    name: "preview-clean",
    role: "user",
  },
  { email: "preview-pro@previews.agpt.co", name: "preview-pro", role: "user" },
  {
    email: "preview-enterprise@previews.agpt.co",
    name: "preview-enterprise",
    role: "user",
  },
] as const;

/**
 * Deterministic user id for FRESH inserts: uuid-shaped truncation of
 * SHA-256(email). Databases seeded by the older SQL (which derived ids with
 * Postgres md5(email)::uuid) are still safe to re-seed: the seeder matches
 * existing identities by email before ever deriving an id, so the
 * derivation only has to be stable, not backward-identical.
 */
export function deterministicUserId(email: string): string {
  const hex = createHash("sha256").update(email).digest("hex").slice(0, 32);
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20, 32),
  ].join("-");
}

/**
 * Schema names come from the AUTH_DB_SCHEMA env var and are interpolated as
 * SQL identifiers, so refuse anything that isn't a plain lowercase
 * identifier rather than attempting to escape it.
 */
export function assertSafeSchemaName(schema: string): string {
  if (!/^[a-z_][a-z0-9_]*$/.test(schema)) {
    throw new Error(`Unsafe schema name: ${JSON.stringify(schema)}`);
  }
  return schema;
}
