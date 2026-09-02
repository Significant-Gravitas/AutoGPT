-- Validate the FK added NOT VALID in 20260727070306_add_expert_entity.
-- Runs in its own transaction so the validation scan holds only a
-- SHARE UPDATE EXCLUSIVE lock (writes proceed); every pre-existing
-- ChatSession row has a NULL expertId, so the scan is trivial.
ALTER TABLE "ChatSession" VALIDATE CONSTRAINT "ChatSession_expertId_fkey";
