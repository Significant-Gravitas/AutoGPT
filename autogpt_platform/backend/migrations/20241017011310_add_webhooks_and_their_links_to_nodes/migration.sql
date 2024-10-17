-- CreateTable
CREATE TABLE "AgentNodeWebhook" (
    "agentNodeId" TEXT NOT NULL,
    "webhookId" TEXT NOT NULL,

    CONSTRAINT "AgentNodeWebhook_pkey" PRIMARY KEY ("agentNodeId","webhookId")
);

-- CreateTable
CREATE TABLE "IntegrationWebhook" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3),
    "credentialsId" TEXT NOT NULL,
    "provider" TEXT NOT NULL,
    "providerWebhookId" TEXT NOT NULL,
    "config" JSONB NOT NULL,
    "token" TEXT NOT NULL,

    CONSTRAINT "IntegrationWebhook_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "AgentNodeWebhook" ADD CONSTRAINT "AgentNodeWebhook_agentNodeId_fkey" FOREIGN KEY ("agentNodeId") REFERENCES "AgentNode"("id") ON DELETE RESTRICT ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "AgentNodeWebhook" ADD CONSTRAINT "AgentNodeWebhook_webhookId_fkey" FOREIGN KEY ("webhookId") REFERENCES "IntegrationWebhook"("id") ON DELETE CASCADE ON UPDATE CASCADE;
