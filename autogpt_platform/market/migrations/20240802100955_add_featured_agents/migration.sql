-- CreateTable
CREATE TABLE "FeaturedAgent" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "agentId" UUID NOT NULL,
    "is_featured" BOOLEAN NOT NULL,
    "category" TEXT NOT NULL DEFAULT 'featured',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "FeaturedAgent_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "FeaturedAgent_id_key" ON "FeaturedAgent"("id");

-- CreateIndex
CREATE UNIQUE INDEX "FeaturedAgent_agentId_key" ON "FeaturedAgent"("agentId");

-- AddForeignKey
ALTER TABLE "FeaturedAgent" ADD CONSTRAINT "FeaturedAgent_agentId_fkey" FOREIGN KEY ("agentId") REFERENCES "Agents"("id") ON DELETE RESTRICT ON UPDATE CASCADE;
