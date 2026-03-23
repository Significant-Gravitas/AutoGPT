-- CreateEnum
CREATE TYPE "TallyComputationStatus" AS ENUM ('PENDING', 'RUNNING', 'READY', 'FAILED');

-- CreateTable
CREATE TABLE "UserInvite" (
    "email" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "userId" TEXT,
    "tallyUnderstanding" JSONB,
    "tallyStatus" "TallyComputationStatus" NOT NULL DEFAULT 'PENDING',
    "tallyComputedAt" TIMESTAMP(3),
    "tallyError" TEXT,

    CONSTRAINT "UserInvite_pkey" PRIMARY KEY ("email")
);

-- CreateIndex
CREATE UNIQUE INDEX "UserInvite_userId_key" ON "UserInvite"("userId");

-- CreateIndex
CREATE INDEX "UserInvite_createdAt_idx" ON "UserInvite"("createdAt");

-- Create the allowed_users table if it does not exist
CREATE TABLE IF NOT EXISTS public.allowed_users (
    email TEXT PRIMARY KEY
);

-- AddForeignKey
ALTER TABLE "UserInvite" ADD CONSTRAINT "UserInvite_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

CREATE OR REPLACE FUNCTION public.create_invite()
RETURNS trigger as $$
BEGIN
    INSERT INTO platform."UserInvite" (email)
    VALUES (NEW.email);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_create_invite
    AFTER INSERT ON public.allowed_users
    FOR EACH ROW
    EXECUTE FUNCTION public.create_invite();
