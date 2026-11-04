-- Background work keeps no clock of its own; only a deadline the launching turn
-- already had is stored. The continuation reads the subtask's chat, so no
-- summary is kept on the record.
ALTER TABLE "ChatBackgroundTask" ALTER COLUMN "deadline" DROP NOT NULL;
ALTER TABLE "ChatBackgroundTask" ADD COLUMN "resultSequence" INTEGER;
-- A task completed before this migration keeps its answer in its child's chat;
-- point at that row the way recovery would, so nothing is lost with the summary.
UPDATE "ChatBackgroundTask" task SET "resultSequence" = (
    SELECT max(message.sequence)
    FROM "ChatMessage" message
    JOIN "ChatMessage" launch ON launch.id = task."childTurnId"
    WHERE message."sessionId" = launch."sessionId"
      AND message.role = 'assistant' AND message.sequence > launch.sequence
      AND NOT EXISTS (
        SELECT 1 FROM "ChatMessage" next
        WHERE next."sessionId" = launch."sessionId" AND next.role = 'user'
          AND next.sequence > launch.sequence AND next.sequence < message.sequence
      )
) WHERE task.outcome IS NOT NULL;
ALTER TABLE "ChatBackgroundTask" DROP COLUMN "result";
