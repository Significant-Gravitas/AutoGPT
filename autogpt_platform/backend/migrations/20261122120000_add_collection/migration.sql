-- Collections: tables the user owns, written by workflows, experts and Otto (flag collections). Additive.

-- CreateTable
CREATE TABLE "Collection" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "nameKey" TEXT NOT NULL,
    "fields" JSONB NOT NULL DEFAULT '[]',
    "keyField" TEXT,
    "revision" INTEGER NOT NULL DEFAULT 1,

    CONSTRAINT "Collection_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "CollectionRow" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "collectionId" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "key" TEXT,
    "writeId" TEXT,
    "data" JSONB NOT NULL DEFAULT '{}',
    "sourceData" JSONB NOT NULL DEFAULT '{}',
    "pending" JSONB NOT NULL DEFAULT '{}',
    "manualFields" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "writtenBy" TEXT NOT NULL,
    "revision" INTEGER NOT NULL DEFAULT 1,

    CONSTRAINT "CollectionRow_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "Collection_userId_nameKey_key" ON "Collection"("userId", "nameKey");

-- CreateIndex
CREATE INDEX "CollectionRow_collectionId_updatedAt_idx" ON "CollectionRow"("collectionId", "updatedAt");

-- CreateIndex
CREATE INDEX "CollectionRow_userId_idx" ON "CollectionRow"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "CollectionRow_collectionId_key_key" ON "CollectionRow"("collectionId", "key");

-- CreateIndex
CREATE UNIQUE INDEX "CollectionRow_collectionId_writeId_key" ON "CollectionRow"("collectionId", "writeId");

-- AddForeignKey
ALTER TABLE "Collection" ADD CONSTRAINT "Collection_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CollectionRow" ADD CONSTRAINT "CollectionRow_collectionId_fkey" FOREIGN KEY ("collectionId") REFERENCES "Collection"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "CollectionRow" ADD CONSTRAINT "CollectionRow_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

