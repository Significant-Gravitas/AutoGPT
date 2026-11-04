-- The user hands work out and never holds a task: tasks assigned to them go to
-- Otto. Nothing is started; Otto works a task once it is assigned again.
UPDATE "Task" SET "assigneeType" = 'otto', "assigneeExpertId" = NULL
WHERE "assigneeType" = 'user';
