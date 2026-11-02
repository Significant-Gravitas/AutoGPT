-- Store review scores are 1-5 stars. The API validates the range; this is the
-- database-side guard (#15304).

-- No UI posts a score outside 1-5, so such a row is not a star rating: drop it
-- rather than clamp it to the extreme it overshot.
DELETE FROM "StoreListingReview" WHERE "score" < 1 OR "score" > 5;

-- AlterTable
ALTER TABLE "StoreListingReview"
    ADD CONSTRAINT "StoreListingReview_score_range" CHECK ("score" BETWEEN 1 AND 5);

-- mv_review_stats caches the averages; recompute them without the dropped rows.
REFRESH MATERIALIZED VIEW "mv_review_stats";
