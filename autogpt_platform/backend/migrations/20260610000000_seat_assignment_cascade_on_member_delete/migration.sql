-- Seat assignments must not block member/org deletion.
--
-- The original FKs were ON DELETE RESTRICT, so deleting a User (which
-- cascades to OrgMember) failed with
-- `OrganizationSeatAssignment_organizationId_userId_fkey` whenever the
-- user held a seat. A seat is a property of the membership: when the
-- membership (or the whole org) goes away, the seat assignment should
-- go with it.

-- DropForeignKey
ALTER TABLE "OrganizationSeatAssignment" DROP CONSTRAINT "OrganizationSeatAssignment_organizationId_userId_fkey";

-- DropForeignKey
ALTER TABLE "OrganizationSeatAssignment" DROP CONSTRAINT "OrganizationSeatAssignment_organizationId_fkey";

-- AddForeignKey
ALTER TABLE "OrganizationSeatAssignment" ADD CONSTRAINT "OrganizationSeatAssignment_organizationId_userId_fkey" FOREIGN KEY ("organizationId", "userId") REFERENCES "OrgMember"("orgId", "userId") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "OrganizationSeatAssignment" ADD CONSTRAINT "OrganizationSeatAssignment_organizationId_fkey" FOREIGN KEY ("organizationId") REFERENCES "Organization"("id") ON DELETE CASCADE ON UPDATE CASCADE;
