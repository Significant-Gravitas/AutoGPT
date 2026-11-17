CREATE TABLE "ChatBackgroundTask" (
    "id" TEXT PRIMARY KEY,
    "userId" TEXT NOT NULL,
    "parentSessionId" TEXT NOT NULL REFERENCES "ChatSession"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    "childSessionId" TEXT NOT NULL UNIQUE REFERENCES "ChatSession"("id") ON DELETE CASCADE ON UPDATE CASCADE,
    "launchTurnId" TEXT NOT NULL,
    "childTurnId" TEXT NOT NULL UNIQUE,
    "context" JSONB NOT NULL,
    "deadline" TIMESTAMP(3) NOT NULL,
    "outcome" TEXT,
    "result" TEXT,
    "startedAt" TIMESTAMP(3),
    "followupStartedAt" TIMESTAMP(3),
    "followupState" TEXT NOT NULL DEFAULT 'waiting',
    "wakeSuppressed" BOOLEAN NOT NULL DEFAULT FALSE,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX "ChatBackgroundTask_followupState_updatedAt_idx" ON "ChatBackgroundTask"("followupState", "updatedAt");
CREATE INDEX "ChatBackgroundTask_userId_parentSessionId_launchTurnId_idx" ON "ChatBackgroundTask"("userId", "parentSessionId", "launchTurnId");
