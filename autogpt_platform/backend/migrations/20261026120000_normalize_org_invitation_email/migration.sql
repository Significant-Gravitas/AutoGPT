-- Invitations now store the email trimmed and lowercased, the form User.email
-- already has, so the invitee's pending list can match it exactly. This
-- normalizes the ones created before that.
UPDATE "OrgInvitation"
SET "email" = lower(btrim("email", E' \t\n\r'))
WHERE "email" <> lower(btrim("email", E' \t\n\r'));
