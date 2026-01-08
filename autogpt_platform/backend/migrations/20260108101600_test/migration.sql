-- CreateTable
CREATE TABLE "Testing" (
    "id" TEXT NOT NULL,
    "maxEmailsPerDay" INTEGER NOT NULL DEFAULT 3,

    CONSTRAINT "Testing_pkey" PRIMARY KEY ("id")
);
