-- Public agent ratings are AVG("score"), so one review with a score outside
-- 1..5 skews the rating for every future read. Bound the column at the
-- storage layer as defense-in-depth behind the API model validation.
--
-- Clamp any rows written before this migration so the constraint cannot fail
-- in environments that already stored out-of-range scores. Clamping keeps the
-- review (and its author/listing linkage) instead of deleting user content.
UPDATE "StoreListingReview"
SET "score" = LEAST(5, GREATEST(1, "score"))
WHERE "score" < 1 OR "score" > 5;

ALTER TABLE "StoreListingReview"
ADD CONSTRAINT "StoreListingReview_score_range"
CHECK ("score" BETWEEN 1 AND 5);
