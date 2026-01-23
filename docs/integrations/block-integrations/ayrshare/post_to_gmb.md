# Ayrshare Post To GMB
<!-- MANUAL: file_description -->
Blocks for posting content to Google My Business profiles using the Ayrshare social media management API.
<!-- END MANUAL -->

## Post To GMB

### What it is
Post to Google My Business using Ayrshare

### How it works
<!-- MANUAL: how_it_works -->
This block uses Ayrshare's API to publish content to Google My Business profiles. It supports standard posts, photo/video posts (categorized by type like exterior, interior, product), and special post types including events and promotional offers with coupon codes.

The block integrates with Google's Business Profile API through Ayrshare, enabling call-to-action buttons (book, order, shop, learn more, sign up, call), event scheduling with start/end dates, and promotional offers with terms and redemption URLs.
<!-- END MANUAL -->

### Inputs

| Input | Description | Type | Required |
|-------|-------------|------|----------|
| post | The post text to be published | str | No |
| media_urls | Optional list of media URLs. GMB supports only one image or video per post. | List[str] | No |
| is_video | Whether the media is a video | bool | No |
| schedule_date | UTC datetime for scheduling (YYYY-MM-DDThh:mm:ssZ) | str (date-time) | No |
| disable_comments | Whether to disable comments | bool | No |
| shorten_links | Whether to shorten links | bool | No |
| unsplash | Unsplash image configuration | str | No |
| requires_approval | Whether to enable approval workflow | bool | No |
| random_post | Whether to generate random post text | bool | No |
| random_media_url | Whether to generate random media | bool | No |
| notes | Additional notes for the post | str | No |
| is_photo_video | Whether this is a photo/video post (appears in Photos section) | bool | No |
| photo_category | Category for photo/video: cover, profile, logo, exterior, interior, product, at_work, food_and_drink, menu, common_area, rooms, teams | str | No |
| call_to_action_type | Type of action button: 'book', 'order', 'shop', 'learn_more', 'sign_up', or 'call' | str | No |
| call_to_action_url | URL for the action button (not required for 'call' action) | str | No |
| event_title | Event title for event posts | str | No |
| event_start_date | Event start date in ISO format (e.g., '2024-03-15T09:00:00Z') | str | No |
| event_end_date | Event end date in ISO format (e.g., '2024-03-15T17:00:00Z') | str | No |
| offer_title | Offer title for promotional posts | str | No |
| offer_start_date | Offer start date in ISO format (e.g., '2024-03-15T00:00:00Z') | str | No |
| offer_end_date | Offer end date in ISO format (e.g., '2024-04-15T23:59:59Z') | str | No |
| offer_coupon_code | Coupon code for the offer (max 58 characters) | str | No |
| offer_redeem_online_url | URL where customers can redeem the offer online | str | No |
| offer_terms_conditions | Terms and conditions for the offer | str | No |

### Outputs

| Output | Description | Type |
|--------|-------------|------|
| error | Error message if the operation failed | str |
| post_result | The result of the post | PostResponse |
| post | The result of the post | PostIds |

### Possible use case
<!-- MANUAL: use_case -->
**Local Business Updates**: Post daily specials, new arrivals, or service announcements directly to your Google Business Profile.

**Promotional Campaigns**: Create time-limited offers with coupon codes and online redemption links to drive sales.

**Event Marketing**: Announce upcoming events with dates, descriptions, and call-to-action buttons for reservations.
<!-- END MANUAL -->

---
