-- A page Otto composes for one library workflow (flag workflow-pages). Additive.

-- CreateTable
CREATE TABLE "WorkflowPage" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "libraryAgentId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "design" JSONB NOT NULL,
    "previousDesign" JSONB,

    CONSTRAINT "WorkflowPage_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "WorkflowPage_libraryAgentId_key" ON "WorkflowPage"("libraryAgentId");

-- AddForeignKey
ALTER TABLE "WorkflowPage" ADD CONSTRAINT "WorkflowPage_libraryAgentId_fkey" FOREIGN KEY ("libraryAgentId") REFERENCES "LibraryAgent"("id") ON DELETE CASCADE ON UPDATE CASCADE;

