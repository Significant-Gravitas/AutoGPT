/*
  Warnings:

  - A unique constraint covering the columns `[userId,type]` on the table `UserNotificationBatch` will be added. If there are existing duplicate values, this will fail.

*/
-- CreateIndex
CREATE UNIQUE INDEX "UserNotificationBatch_userId_type_key" ON "UserNotificationBatch"("userId", "type");
