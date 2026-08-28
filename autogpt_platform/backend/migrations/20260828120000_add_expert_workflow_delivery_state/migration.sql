CREATE TYPE "ExpertWorkflowValidationStatus" AS ENUM ('PASSED', 'FAILED');

CREATE TYPE "ExpertWorkflowDeliveryTarget" AS ENUM ('MESSAGE', 'WORKSPACE_FILES');

CREATE TYPE "ExpertWorkflowDeliveryStatus" AS ENUM (
  'QUEUED',
  'RUNNING',
  'DELIVERED',
  'PARTIAL',
  'BLOCKED',
  'FAILED'
);

ALTER TABLE "ExpertWorkflow"
  ADD COLUMN "validationGraphVersion" INTEGER,
  ADD COLUMN "validationExecutionId" TEXT,
  ADD COLUMN "deliveryTarget" "ExpertWorkflowDeliveryTarget" NOT NULL DEFAULT 'MESSAGE',
  ADD COLUMN "artifactOutputNames" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];

CREATE TABLE "ExpertWorkflowValidation" (
  "id" TEXT NOT NULL,
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "userId" TEXT NOT NULL,
  "libraryAgentId" TEXT NOT NULL,
  "testExecutionId" TEXT NOT NULL,
  "sessionId" TEXT NOT NULL,
  "graphId" TEXT NOT NULL,
  "graphVersion" INTEGER NOT NULL,
  "status" "ExpertWorkflowValidationStatus" NOT NULL,
  "deliveryTarget" "ExpertWorkflowDeliveryTarget" NOT NULL DEFAULT 'MESSAGE',
  "artifactOutputNames" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  "artifacts" JSONB NOT NULL DEFAULT '[]',
  "requiredArtifactsPresent" BOOLEAN NOT NULL DEFAULT true,
  "nodeErrorCount" INTEGER NOT NULL DEFAULT 0,
  "nodeFailures" JSONB NOT NULL DEFAULT '[]',
  CONSTRAINT "ExpertWorkflowValidation_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "ExpertWorkflowValidation_testExecutionId_key"
  ON "ExpertWorkflowValidation"("testExecutionId");
CREATE INDEX "ExpertWorkflowValidation_lookup_idx"
  ON "ExpertWorkflowValidation"(
    "userId", "libraryAgentId", "graphId", "graphVersion", "status", "createdAt"
  );

ALTER TABLE "ExpertWorkflowValidation"
  ADD CONSTRAINT "ExpertWorkflowValidation_userId_fkey"
  FOREIGN KEY ("userId") REFERENCES "User"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ExpertWorkflowValidation"
  ADD CONSTRAINT "ExpertWorkflowValidation_libraryAgentId_fkey"
  FOREIGN KEY ("libraryAgentId") REFERENCES "LibraryAgent"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ExpertWorkflowValidation"
  ADD CONSTRAINT "ExpertWorkflowValidation_testExecutionId_fkey"
  FOREIGN KEY ("testExecutionId") REFERENCES "AgentGraphExecution"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;

CREATE TABLE "ExpertWorkflowRunState" (
  "id" TEXT NOT NULL,
  "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "completedAt" TIMESTAMP(3),
  "userId" TEXT NOT NULL,
  "expertId" TEXT NOT NULL,
  "workflowId" TEXT NOT NULL,
  "executionId" TEXT NOT NULL,
  "status" "ExpertWorkflowDeliveryStatus" NOT NULL DEFAULT 'QUEUED',
  "deliveryTarget" "ExpertWorkflowDeliveryTarget" NOT NULL DEFAULT 'MESSAGE',
  "artifactOutputNames" TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  "artifacts" JSONB NOT NULL DEFAULT '[]',
  "requiredArtifactsPresent" BOOLEAN NOT NULL DEFAULT true,
  "nodeErrorCount" INTEGER NOT NULL DEFAULT 0,
  "nodeFailures" JSONB NOT NULL DEFAULT '[]',
  "blocker" TEXT,
  CONSTRAINT "ExpertWorkflowRunState_pkey" PRIMARY KEY ("id")
);

CREATE UNIQUE INDEX "ExpertWorkflowRunState_executionId_key"
  ON "ExpertWorkflowRunState"("executionId");
CREATE INDEX "ExpertWorkflowRunState_userId_expertId_status_createdAt_idx"
  ON "ExpertWorkflowRunState"("userId", "expertId", "status", "createdAt");
CREATE INDEX "ExpertWorkflowRunState_workflowId_createdAt_idx"
  ON "ExpertWorkflowRunState"("workflowId", "createdAt");

ALTER TABLE "ExpertWorkflowRunState"
  ADD CONSTRAINT "ExpertWorkflowRunState_userId_fkey"
  FOREIGN KEY ("userId") REFERENCES "User"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ExpertWorkflowRunState"
  ADD CONSTRAINT "ExpertWorkflowRunState_expertId_fkey"
  FOREIGN KEY ("expertId") REFERENCES "Expert"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ExpertWorkflowRunState"
  ADD CONSTRAINT "ExpertWorkflowRunState_workflowId_fkey"
  FOREIGN KEY ("workflowId") REFERENCES "ExpertWorkflow"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
ALTER TABLE "ExpertWorkflowRunState"
  ADD CONSTRAINT "ExpertWorkflowRunState_executionId_fkey"
  FOREIGN KEY ("executionId") REFERENCES "AgentGraphExecution"("id")
  ON DELETE CASCADE ON UPDATE CASCADE;
