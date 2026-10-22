-- The usage attribution (#15208) a dream pass started under: the trial or
-- paid generation its phases are charged to, captured once when the pass
-- starts. Every path that charges a batch pass's landed phases reads it here,
-- whatever process runs it and however late: the result callbacks, a failed
-- pass's end, the drop of a closed pass's batch and the reaper's cleanup. A
-- row without one is never charged: one written before this column, or one
-- whose capture failed. A nullable column on an existing table; no row is
-- rewritten.

-- AlterTable
ALTER TABLE "DreamPass" ADD COLUMN     "costContext" JSONB;
