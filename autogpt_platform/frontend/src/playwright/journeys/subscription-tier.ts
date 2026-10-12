import { Client } from "pg";

/**
 * Gives a journey user a paid plan without Stripe, so the hosted paywall lets
 * them run agents. A paid tier with no Stripe customer is how the platform
 * models an admin grant, which the Stripe reconciliation sweep never revokes.
 *
 * Opt in with PLAYWRIGHT_JOURNEY_TIER (e.g. PRO). It needs the stack's
 * database: PLAYWRIGHT_JOURNEY_DATABASE_URL, else DATABASE_URL from
 * frontend/.env. Unset on a preview, where an existing paid account
 * (PLAYWRIGHT_JOURNEY_EMAIL) is the way past the paywall.
 */
export async function grantSubscriptionTier(userID: string) {
  const tier = process.env.PLAYWRIGHT_JOURNEY_TIER;
  if (!tier) return;
  const connectionString =
    process.env.PLAYWRIGHT_JOURNEY_DATABASE_URL ?? process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error(
      "PLAYWRIGHT_JOURNEY_TIER needs PLAYWRIGHT_JOURNEY_DATABASE_URL or DATABASE_URL",
    );
  }
  const client = new Client({ connectionString });
  await client.connect();
  try {
    const result = await client.query(
      `UPDATE platform."User" SET "subscriptionTier" = $1::platform."SubscriptionTier" WHERE id = $2`,
      [tier, userID],
    );
    if (result.rowCount !== 1) {
      throw new Error(`no platform user ${userID} to grant ${tier} to`);
    }
  } finally {
    await client.end();
  }
}
