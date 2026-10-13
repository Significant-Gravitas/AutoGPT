CREATE TABLE "ChatUIDraft" (
    "sessionId" TEXT NOT NULL,
    "viewId" TEXT NOT NULL,
    "toolCallId" TEXT NOT NULL,
    "sequence" INTEGER NOT NULL,
    "source" TEXT NOT NULL,
    "state" JSONB NOT NULL,
    "revision" INTEGER NOT NULL DEFAULT 1,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "ChatUIDraft_pkey" PRIMARY KEY ("sessionId", "viewId"),
    CONSTRAINT "ChatUIDraft_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "ChatSession"("id") ON DELETE CASCADE ON UPDATE CASCADE
);
