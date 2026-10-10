-- The account each turn of a routine uses where the owner has several for one
-- provider, chosen in the chat that set it up (SECRT-2804).
ALTER TABLE "ExpertRoutine" ADD COLUMN "credentialPins" JSONB;
