-- AlterTable
ALTER TABLE "platform"."LibraryAgent" ADD COLUMN "topIntegrations" JSONB NOT NULL DEFAULT '[]';
