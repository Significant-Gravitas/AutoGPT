/*
  Warnings:

  - You are about to drop the column `blockId` on the `CreditTransaction` table. All the data in the column will be moved to metadata->block_id.

*/
BEGIN;

-- DropForeignKey blockId
ALTER TABLE "CreditTransaction" DROP CONSTRAINT "CreditTransaction_blockId_fkey";

-- Update migrate blockId into metadata->"block_id"
UPDATE "CreditTransaction" SET "metadata" = jsonb_set("metadata"::jsonb, '{block_id}', to_jsonb("blockId")) WHERE "blockId" IS NOT NULL;

-- AlterTable drop blockId
ALTER TABLE "CreditTransaction" DROP COLUMN "blockId";

-- CreateIndex graphExecId
CREATE INDEX "CreditTransaction_graphExecId_idx" ON "CreditTransaction"(("metadata"->>'graph_exec_id'));

COMMIT;
