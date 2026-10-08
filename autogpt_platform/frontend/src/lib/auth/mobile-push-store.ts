import type { Pool } from "pg";
import type { MobilePushStore } from "./mobile-push";

export function mobilePushStore(pool: Pool): MobilePushStore {
  return {
    async save(registration) {
      const client = await pool.connect();
      try {
        await client.query("BEGIN");
        const session = await client.query(
          'SELECT "id" FROM "UserAuthSession" WHERE "id" = $1 AND "expiresAt" > NOW() FOR UPDATE',
          [registration.sessionID],
        );
        if (session.rowCount !== 1) throw new Error("Mobile session expired");
        await client.query(
          'DELETE FROM "NativePushSubscription" WHERE "sessionId" = $1',
          [registration.sessionID],
        );
        await client.query(
          `INSERT INTO "NativePushSubscription" ("id", "sessionId", "provider", "token", "environment", "origin", "updatedAt")
           VALUES ($1, $2, $3, $4, $5, $6, NOW())
           ON CONFLICT ("provider", "token", "environment") DO UPDATE SET
             "id" = EXCLUDED."id", "sessionId" = EXCLUDED."sessionId", "origin" = EXCLUDED."origin", "updatedAt" = NOW()`,
          [
            registration.binding_id,
            registration.sessionID,
            registration.provider,
            registration.token,
            registration.environment,
            registration.origin,
          ],
        );
        await client.query("COMMIT");
      } catch (error) {
        await client.query("ROLLBACK");
        throw error;
      } finally {
        client.release();
      }
    },
    async remove(sessionID, bindingID) {
      await pool.query(
        'DELETE FROM "NativePushSubscription" WHERE "sessionId" = $1 AND ($2::text IS NULL OR "id" = $2)',
        [sessionID, bindingID ?? null],
      );
    },
  };
}
