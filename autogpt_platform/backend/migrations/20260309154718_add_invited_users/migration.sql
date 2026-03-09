-- CreateEnum
CREATE TYPE "InvitedUserStatus" AS ENUM ('INVITED', 'CLAIMED', 'REVOKED');

-- CreateEnum
CREATE TYPE "TallyComputationStatus" AS ENUM ('PENDING', 'RUNNING', 'READY', 'FAILED');

-- CreateTable
CREATE TABLE "InvitedUser" (
    "id" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,
    "email" TEXT NOT NULL,
    "status" "InvitedUserStatus" NOT NULL DEFAULT 'INVITED',
    "authUserId" TEXT,
    "name" TEXT,
    "tallyUnderstanding" JSONB,
    "tallyStatus" "TallyComputationStatus" NOT NULL DEFAULT 'PENDING',
    "tallyComputedAt" TIMESTAMP(3),
    "tallyError" TEXT,

    CONSTRAINT "InvitedUser_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "InvitedUser_email_key" ON "InvitedUser"("email");

-- CreateIndex
CREATE UNIQUE INDEX "InvitedUser_authUserId_key" ON "InvitedUser"("authUserId");

-- CreateIndex
CREATE INDEX "InvitedUser_status_idx" ON "InvitedUser"("status");

-- CreateIndex
CREATE INDEX "InvitedUser_tallyStatus_idx" ON "InvitedUser"("tallyStatus");

-- AddForeignKey
ALTER TABLE "InvitedUser" ADD CONSTRAINT "InvitedUser_authUserId_fkey"
FOREIGN KEY ("authUserId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

DO $$
DECLARE
    allowed_users_schema TEXT;
BEGIN
    SELECT table_schema
      INTO allowed_users_schema
      FROM information_schema.columns
     WHERE table_name = 'allowed_users'
       AND column_name = 'email'
     ORDER BY CASE
        WHEN table_schema = 'platform' THEN 0
        WHEN table_schema = 'public' THEN 1
        ELSE 2
     END
     LIMIT 1;

    IF allowed_users_schema IS NOT NULL THEN
        EXECUTE format(
            'INSERT INTO platform."InvitedUser" ("id", "email", "status", "tallyStatus", "createdAt", "updatedAt")
             SELECT gen_random_uuid()::text,
                    lower(email),
                    ''INVITED''::platform."InvitedUserStatus",
                    ''PENDING''::platform."TallyComputationStatus",
                    now(),
                    now()
               FROM %I.allowed_users
              WHERE email IS NOT NULL
             ON CONFLICT ("email") DO NOTHING',
            allowed_users_schema
        );
    END IF;
END $$;

CREATE OR REPLACE FUNCTION platform.ensure_invited_user_can_register()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.email IS NULL THEN
        RAISE EXCEPTION 'The email address "%" is not allowed to register. Please contact support for assistance.', NEW.email
            USING ERRCODE = 'P0001';
    END IF;

    IF lower(split_part(NEW.email, '@', 2)) = 'agpt.co' THEN
        RETURN NEW;
    END IF;

    IF EXISTS (
        SELECT 1
          FROM platform."InvitedUser" invited_user
         WHERE lower(invited_user.email) = lower(NEW.email)
           AND invited_user.status = 'INVITED'::platform."InvitedUserStatus"
    ) THEN
        RETURN NEW;
    END IF;

    RAISE EXCEPTION 'The email address "%" is not allowed to register. Please contact support for assistance.', NEW.email
        USING ERRCODE = 'P0001';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM information_schema.tables
         WHERE table_schema = 'auth'
           AND table_name = 'users'
    ) THEN
        DROP TRIGGER IF EXISTS user_added_to_platform ON auth.users;
        DROP TRIGGER IF EXISTS invited_user_signup_gate ON auth.users;

        CREATE TRIGGER invited_user_signup_gate
        BEFORE INSERT ON auth.users
        FOR EACH ROW EXECUTE FUNCTION platform.ensure_invited_user_can_register();
    END IF;
END $$;
