import { Client } from "pg";

/**
 * Runs `query` against the stack's database, for what a journey can't do or
 * read through the app: granting a plan, reading the cost ledger.
 *
 * Uses PLAYWRIGHT_JOURNEY_DATABASE_URL, else DATABASE_URL from frontend/.env.
 * Unset on a preview, which journeys can't reach into.
 */
export async function withJourneyDatabase<T>(
  purpose: string,
  query: (client: Client) => Promise<T>,
) {
  const connectionString =
    process.env.PLAYWRIGHT_JOURNEY_DATABASE_URL ?? process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error(
      `${purpose} needs PLAYWRIGHT_JOURNEY_DATABASE_URL or DATABASE_URL`,
    );
  }
  const client = new Client({ connectionString });
  await client.connect();
  try {
    return await query(client);
  } finally {
    await client.end();
  }
}
