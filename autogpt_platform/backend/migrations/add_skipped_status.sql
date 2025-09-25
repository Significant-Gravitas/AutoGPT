-- Migration: Add SKIPPED status to AgentExecutionStatus enum
-- This migration adds support for conditional/optional block execution

-- Add SKIPPED value to the AgentExecutionStatus enum
ALTER TYPE "AgentExecutionStatus" ADD VALUE 'SKIPPED';

-- Note: This migration is irreversible in PostgreSQL.
-- Enum values cannot be removed once added.
-- To run this migration, execute:
-- cd autogpt_platform/backend && poetry run prisma migrate dev --name add-skipped-execution-status