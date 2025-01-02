/*
  Warnings:

  - You are about to drop the column `category` on the `FeaturedAgent` table. All the data in the column will be lost.
  - You are about to drop the column `is_featured` on the `FeaturedAgent` table. All the data in the column will be lost.

*/
-- AlterTable
ALTER TABLE "FeaturedAgent" DROP COLUMN "category",
DROP COLUMN "is_featured",
ADD COLUMN     "featuredCategories" TEXT[],
ADD COLUMN     "isActive" BOOLEAN NOT NULL DEFAULT false;
