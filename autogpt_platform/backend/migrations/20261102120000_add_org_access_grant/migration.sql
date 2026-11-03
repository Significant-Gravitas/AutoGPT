-- An admin's grant of organizations to one account (Admin -> Organization
-- Access). The account's team organizations pass it on to their members and
-- invitees; see backend/data/org_access.py.

-- CreateTable
CREATE TABLE "OrgAccessGrant" (
    "userId" TEXT NOT NULL,
    "grantedByUserId" TEXT NOT NULL,
    "note" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "OrgAccessGrant_pkey" PRIMARY KEY ("userId")
);

-- AddForeignKey
ALTER TABLE "OrgAccessGrant" ADD CONSTRAINT "OrgAccessGrant_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
