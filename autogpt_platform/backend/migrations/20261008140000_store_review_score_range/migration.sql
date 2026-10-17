-- Store review scores are 1-5 stars, but nothing enforced it, so one request
-- could post score=1000000 and skew a listing's AVG(score) rating (#15304).
-- The API now validates the range; this is the DB-side guard.

-- Out-of-range rows can only have come from bypassing the UI; drop them so the
-- constraint can be added.
DELETE FROM "StoreListingReview" WHERE "score" < 1 OR "score" > 5;

-- AlterTable
ALTER TABLE "StoreListingReview"
    ADD CONSTRAINT "StoreListingReview_score_range" CHECK ("score" BETWEEN 1 AND 5);

-- Ratings are pre-aggregated; recompute them without the deleted rows.
REFRESH MATERIALIZED VIEW "mv_review_stats";
