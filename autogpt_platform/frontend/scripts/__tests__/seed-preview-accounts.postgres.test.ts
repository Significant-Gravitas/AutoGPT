import { randomUUID } from "node:crypto";
import { Pool } from "pg";
import { describe, expect, it } from "vitest";
import { PREVIEW_ACCOUNTS, seedRoster } from "../seed-preview-accounts.helpers";

const connectionString = process.env.PREVIEW_ACCOUNTS_TEST_DATABASE_URL;
const describePostgres = connectionString ? describe : describe.skip;

describePostgres("preview account seeding against PostgreSQL", () => {
  it("is idempotent, converges onboarding, and preserves Stripe-owned tiers", async () => {
    const pool = new Pool({ connectionString });
    const schema = `preview_seed_${randomUUID().replaceAll("-", "")}`;
    function qualified(name: string) {
      return `"${schema}"."${name}"`;
    }
    const tables = {
      identityTable: qualified("UserAuthIdentity"),
      accountTable: qualified("UserAuthAccount"),
      userTable: qualified("User"),
      profileTable: qualified("Profile"),
      onboardingTable: qualified("UserOnboarding"),
      subscriptionTierType: qualified("SubscriptionTier"),
      onboardingStepType: qualified("OnboardingStep"),
      passwordHash: "$2b$10$previewintegrationtesthash",
    };

    try {
      await pool.query(`CREATE SCHEMA "${schema}"`);
      await pool.query(`
          CREATE TYPE ${tables.subscriptionTierType} AS ENUM
            ('NO_TIER', 'BASIC', 'PRO', 'MAX', 'BUSINESS', 'ENTERPRISE');
          CREATE TYPE ${tables.onboardingStepType} AS ENUM
            ('WELCOME', 'VISIT_COPILOT');

          CREATE TABLE ${tables.identityTable} (
            id text PRIMARY KEY,
            name text NOT NULL,
            email text NOT NULL UNIQUE,
            "emailVerified" boolean NOT NULL DEFAULT false,
            role text,
            "createdAt" timestamptz NOT NULL DEFAULT now(),
            "updatedAt" timestamptz NOT NULL DEFAULT now()
          );

          CREATE TABLE ${tables.accountTable} (
            id text PRIMARY KEY,
            "accountId" text NOT NULL,
            "providerId" text NOT NULL,
            "userId" text NOT NULL REFERENCES ${tables.identityTable}(id),
            password text,
            "createdAt" timestamptz NOT NULL DEFAULT now(),
            "updatedAt" timestamptz NOT NULL DEFAULT now()
          );

          CREATE TABLE ${tables.userTable} (
            id text PRIMARY KEY,
            email text NOT NULL UNIQUE,
            "emailVerified" boolean NOT NULL DEFAULT true,
            name text,
            "stripeCustomerId" text,
            "subscriptionTier" ${tables.subscriptionTierType} NOT NULL DEFAULT 'NO_TIER',
            "createdAt" timestamptz NOT NULL DEFAULT now(),
            "updatedAt" timestamptz NOT NULL DEFAULT now()
          );

          CREATE TABLE ${tables.profileTable} (
            id text PRIMARY KEY,
            "userId" text NOT NULL UNIQUE REFERENCES ${tables.userTable}(id),
            name text NOT NULL,
            username text NOT NULL UNIQUE,
            description text NOT NULL,
            links text[] NOT NULL DEFAULT ARRAY[]::text[],
            "avatarUrl" text,
            "createdAt" timestamptz NOT NULL DEFAULT now(),
            "updatedAt" timestamptz NOT NULL DEFAULT now()
          );

          CREATE TABLE ${tables.onboardingTable} (
            id text PRIMARY KEY,
            "userId" text NOT NULL UNIQUE REFERENCES ${tables.userTable}(id),
            "completedSteps" ${tables.onboardingStepType}[] NOT NULL DEFAULT ARRAY[]::${tables.onboardingStepType}[],
            "createdAt" timestamptz NOT NULL DEFAULT now(),
            "updatedAt" timestamptz DEFAULT now()
          );
        `);

      const fresh = await seedRoster(pool, tables);
      expect(fresh).toEqual({
        createdIdentities: 5,
        createdAccounts: 5,
        createdUsers: 5,
        updatedUsers: 0,
        createdProfiles: 5,
        changedOnboarding: 5,
      });

      const productStates = await pool.query<{
        email: string;
        subscriptionTier: string;
        stripeCustomerId: string | null;
        onboardingComplete: boolean;
      }>(`
          SELECT
            users.email,
            users."subscriptionTier"::text AS "subscriptionTier",
            users."stripeCustomerId" AS "stripeCustomerId",
            'VISIT_COPILOT'::${tables.onboardingStepType}
              = ANY(onboarding."completedSteps") AS "onboardingComplete"
          FROM ${tables.userTable} AS users
          JOIN ${tables.onboardingTable} AS onboarding
            ON onboarding."userId" = users.id
          ORDER BY users.email
        `);
      expect(productStates.rows).toEqual(
        [...PREVIEW_ACCOUNTS]
          .sort((left, right) => left.email.localeCompare(right.email))
          .map((account) => ({
            email: account.email,
            subscriptionTier: account.subscriptionTier,
            stripeCustomerId: null,
            onboardingComplete: account.onboardingComplete,
          })),
      );

      await expect(seedRoster(pool, tables)).resolves.toEqual({
        createdIdentities: 0,
        createdAccounts: 0,
        createdUsers: 0,
        updatedUsers: 0,
        createdProfiles: 0,
        changedOnboarding: 0,
      });

      await pool.query(
        `UPDATE ${tables.userTable}
           SET "stripeCustomerId" = 'cus_preview_preserve_test',
               "subscriptionTier" = 'BASIC'
           WHERE email = 'preview-pro@previews.agpt.co';

           UPDATE ${tables.userTable}
           SET "subscriptionTier" = 'BASIC'
           WHERE email = 'preview-enterprise@previews.agpt.co';

           UPDATE ${tables.onboardingTable} AS onboarding
           SET "completedSteps" = ARRAY['WELCOME']::${tables.onboardingStepType}[]
           FROM ${tables.userTable} AS users
           WHERE onboarding."userId" = users.id
             AND users.email = 'preview-existing@previews.agpt.co';

           UPDATE ${tables.onboardingTable} AS onboarding
           SET "completedSteps" = ARRAY['WELCOME', 'VISIT_COPILOT']::${tables.onboardingStepType}[]
           FROM ${tables.userTable} AS users
           WHERE onboarding."userId" = users.id
             AND users.email = 'preview-clean@previews.agpt.co'`,
      );

      await expect(seedRoster(pool, tables)).resolves.toEqual({
        createdIdentities: 0,
        createdAccounts: 0,
        createdUsers: 0,
        updatedUsers: 1,
        createdProfiles: 0,
        changedOnboarding: 2,
      });

      const converged = await pool.query<{
        email: string;
        subscriptionTier: string;
        stripeCustomerId: string | null;
        completedSteps: string[];
      }>(`
          SELECT
            users.email,
            users."subscriptionTier"::text AS "subscriptionTier",
            users."stripeCustomerId" AS "stripeCustomerId",
            onboarding."completedSteps"::text[] AS "completedSteps"
          FROM ${tables.userTable} AS users
          JOIN ${tables.onboardingTable} AS onboarding
            ON onboarding."userId" = users.id
          WHERE users.email IN (
            'preview-pro@previews.agpt.co',
            'preview-enterprise@previews.agpt.co',
            'preview-existing@previews.agpt.co',
            'preview-clean@previews.agpt.co'
          )
          ORDER BY users.email
        `);
      expect(converged.rows).toEqual([
        {
          email: "preview-clean@previews.agpt.co",
          subscriptionTier: "NO_TIER",
          stripeCustomerId: null,
          completedSteps: ["WELCOME"],
        },
        {
          email: "preview-enterprise@previews.agpt.co",
          subscriptionTier: "ENTERPRISE",
          stripeCustomerId: null,
          completedSteps: ["VISIT_COPILOT"],
        },
        {
          email: "preview-existing@previews.agpt.co",
          subscriptionTier: "NO_TIER",
          stripeCustomerId: null,
          completedSteps: ["WELCOME", "VISIT_COPILOT"],
        },
        {
          email: "preview-pro@previews.agpt.co",
          subscriptionTier: "BASIC",
          stripeCustomerId: "cus_preview_preserve_test",
          completedSteps: ["VISIT_COPILOT"],
        },
      ]);

      await expect(seedRoster(pool, tables)).resolves.toEqual({
        createdIdentities: 0,
        createdAccounts: 0,
        createdUsers: 0,
        updatedUsers: 0,
        createdProfiles: 0,
        changedOnboarding: 0,
      });
    } finally {
      try {
        await pool.query(`DROP SCHEMA IF EXISTS "${schema}" CASCADE`);
      } finally {
        await pool.end();
      }
    }
  }, 20_000);
});
