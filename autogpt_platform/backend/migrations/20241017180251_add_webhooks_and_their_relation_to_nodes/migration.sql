-- AlterTable
ALTER TABLE "AgentNode" ADD COLUMN     "webhookId" TEXT;

-- CreateTable
CREATE TABLE "IntegrationWebhook" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3),
    "userId" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "credentialsId" TEXT NOT NULL,
    "webhookType" TEXT NOT NULL,
    "resource" TEXT NOT NULL,
    "events" TEXT[],
    "config" JSONB NOT NULL,
    "secret" TEXT NOT NULL,
    "providerWebhookId" TEXT NOT NULL,

    CONSTRAINT "IntegrationWebhook_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "AgentNode" ADD CONSTRAINT "AgentNode_webhookId_fkey" FOREIGN KEY ("webhookId") REFERENCES "IntegrationWebhook"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "IntegrationWebhook" ADD CONSTRAINT "IntegrationWebhook_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
