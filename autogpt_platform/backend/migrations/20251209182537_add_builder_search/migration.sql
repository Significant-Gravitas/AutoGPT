-- Create BuilderSearchHistory table
CREATE TABLE "BuilderSearchHistory" (
    "id" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "searchQuery" TEXT NOT NULL,
    "filter" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "byCreator" TEXT[] DEFAULT ARRAY[]::TEXT[],

    CONSTRAINT "BuilderSearchHistory_pkey" PRIMARY KEY ("id")
);

-- Define User foreign relation
ALTER TABLE "BuilderSearchHistory" ADD CONSTRAINT "BuilderSearchHistory_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
