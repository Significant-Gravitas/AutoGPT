-- Add imageUrl column
ALTER TABLE "LibraryAgent"
ADD COLUMN  "creatorId" TEXT,
ADD COLUMN  "imageUrl"  TEXT;

-- Add foreign key constraint for creatorId -> Profile
ALTER TABLE "LibraryAgent"
ADD CONSTRAINT "LibraryAgent_creatorId_fkey" FOREIGN KEY ("creatorId") REFERENCES "Profile"("id") ON DELETE SET NULL ON UPDATE CASCADE;
