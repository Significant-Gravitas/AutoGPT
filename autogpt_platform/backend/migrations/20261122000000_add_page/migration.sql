-- Pages Otto composes from the user's workflows and experts (flag pages). Additive.

-- CreateTable
CREATE TABLE "Page" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "title" TEXT NOT NULL,
    "design" JSONB NOT NULL,
    "previousDesign" JSONB,
    "revision" INTEGER NOT NULL DEFAULT 1,
    "pinnedLibraryAgentId" TEXT,
    "chatSessionId" TEXT,

    CONSTRAINT "Page_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Page_pinnedLibraryAgentId_key" ON "Page"("pinnedLibraryAgentId");

-- CreateIndex
CREATE INDEX "Page_userId_idx" ON "Page"("userId");

-- AddForeignKey
ALTER TABLE "Page" ADD CONSTRAINT "Page_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Page" ADD CONSTRAINT "Page_pinnedLibraryAgentId_fkey" FOREIGN KEY ("pinnedLibraryAgentId") REFERENCES "LibraryAgent"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Page" ADD CONSTRAINT "Page_chatSessionId_fkey" FOREIGN KEY ("chatSessionId") REFERENCES "ChatSession"("id") ON DELETE SET NULL ON UPDATE CASCADE;

