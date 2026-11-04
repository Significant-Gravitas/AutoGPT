-- Background work keeps no clock of its own; only a deadline the launching turn
-- already had is stored. The continuation reads the subtask's chat, so no
-- summary is kept on the record.
ALTER TABLE "ChatBackgroundTask" ALTER COLUMN "deadline" DROP NOT NULL;
ALTER TABLE "ChatBackgroundTask" DROP COLUMN "result";
ALTER TABLE "ChatBackgroundTask" ADD COLUMN "resultSequence" INTEGER;
