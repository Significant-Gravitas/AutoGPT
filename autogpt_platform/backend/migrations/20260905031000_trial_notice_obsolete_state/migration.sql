ALTER TABLE "TrialNotificationDelivery"
    DROP CONSTRAINT "TrialNotificationDelivery_status_check",
    ADD CONSTRAINT "TrialNotificationDelivery_status_check"
    CHECK ("status" IN ('pending', 'sending', 'accepted', 'suppressed', 'obsolete', 'failed'));
