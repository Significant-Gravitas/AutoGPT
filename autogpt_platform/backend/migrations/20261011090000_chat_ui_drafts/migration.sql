CREATE TABLE "ChatUIDraft" (
    "id" TEXT NOT NULL,
    "sessionId" TEXT,
    "viewId" TEXT NOT NULL,
    "fileId" TEXT,
    "toolCallId" TEXT,
    "sequence" INTEGER NOT NULL,
    "source" TEXT NOT NULL,
    "state" JSONB NOT NULL,
    "revision" INTEGER NOT NULL DEFAULT 1,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "ChatUIDraft_pkey" PRIMARY KEY ("id"),
    -- An unsaved view's edits belong to its chat, a saved view's to its file.
    CONSTRAINT "ChatUIDraft_one_owner_check" CHECK (("sessionId" IS NULL) <> ("fileId" IS NULL))
);

CREATE UNIQUE INDEX "ChatUIDraft_fileId_key" ON "ChatUIDraft"("fileId");

CREATE UNIQUE INDEX "ChatUIDraft_sessionId_viewId_key" ON "ChatUIDraft"("sessionId", "viewId");

ALTER TABLE "ChatUIDraft" ADD CONSTRAINT "ChatUIDraft_sessionId_fkey" FOREIGN KEY ("sessionId") REFERENCES "ChatSession"("id") ON DELETE CASCADE ON UPDATE CASCADE;

ALTER TABLE "ChatUIDraft" ADD CONSTRAINT "ChatUIDraft_fileId_fkey" FOREIGN KEY ("fileId") REFERENCES "UserWorkspaceFile"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- Finds the live file that holds a view (find_openui_view_file).
CREATE INDEX "UserWorkspaceFile_openui_viewId_idx" ON "UserWorkspaceFile"("workspaceId", (metadata->'openui'->>'viewId')) WHERE "isDeleted" = false AND metadata ? 'openui';
