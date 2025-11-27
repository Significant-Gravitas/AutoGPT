-- CreateTable
CREATE TABLE "BuilderSearch" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "searchQuery" TEXT NOT NULL,
    "filter" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "byCreator" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "userId" TEXT NOT NULL,

    CONSTRAINT "BuilderSearch_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "BuilderSearch" ADD CONSTRAINT "BuilderSearch_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
