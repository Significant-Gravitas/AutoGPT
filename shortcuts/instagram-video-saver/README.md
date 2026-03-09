# Claude-InstaSaver

**An iPhone Shortcut that downloads Instagram videos directly to your Photos library.**

> **Disclaimer**: This shortcut is intended for personal, educational use only. Downloading
> Instagram content may violate Instagram's [Terms of Use](https://help.instagram.com/581066165581870).
> Only download videos you have the right to save (your own content or content with explicit permission).
> See [ISSUES.md](ISSUES.md) for full details.

---

## Features

- Save Instagram **Reels**, **Posts**, and **IGTV** videos to your Camera Roll
- Works from the **iOS Share Sheet** (tap Share → Claude-InstaSaver)
- Works **standalone** — just open and paste a URL
- **Two-method extraction** with automatic fallback for reliability
- Clear **success / error feedback** for every attempt
- No account login required for public posts
- No data stored or transmitted beyond what's needed for the download

## Requirements

| Requirement | Minimum |
|-------------|---------|
| iOS / iPadOS | 15.0+ |
| Shortcuts app | Built-in (iOS 13+) |
| Internet connection | Required |
| Photos permission | Required (prompted on first use) |

---

## Installation

### Method 1 — Import the `.shortcut` File (Recommended)

1. Download `Claude-InstaSaver.shortcut` from this repository to your iPhone/iPad.
2. Open the **Files** app and locate the downloaded file.
3. Tap the file — the **Shortcuts** app opens automatically.
4. Tap **Add Shortcut** (or **Add Untrusted Shortcut** if prompted — see note below).
5. Done! The shortcut appears in your Shortcuts library.

> **"Untrusted Shortcut" prompt**: Go to **Settings → Shortcuts** and enable
> **Allow Untrusted Shortcuts** if prompted. This is normal for shortcuts not
> distributed through the Shortcuts Gallery.

### Method 2 — AirDrop

1. On a Mac/iPhone that already has the file, AirDrop `Claude-InstaSaver.shortcut` to your device.
2. Accept the incoming file — Shortcuts opens and prompts you to add it.
3. Tap **Add Shortcut**.

### Method 3 — Build Manually from Source

If you want to inspect or customise every action before importing:

1. Install Python 3.6+ on your Mac.
2. Clone this repo and navigate to `shortcuts/instagram-video-saver/`.
3. Run `python3 generate_shortcut.py` — this produces `Claude-InstaSaver.shortcut`.
4. Transfer the file to your iPhone via AirDrop or iCloud Drive and follow Method 1.

---

## Usage

### Share Sheet (most common)

1. Open the Instagram app and find the video you want to save.
2. Tap the **⋯ (more)** button → **Share** → **Copy Link**.
3. Open **Safari** and paste the link, or tap **Share** directly from Instagram.
4. In the Share Sheet, scroll down and tap **Claude-InstaSaver**.
5. The shortcut runs, downloads the video, and saves it to your Camera Roll.
6. You'll receive a notification: **"✅ Instagram video saved to Photos!"**

### Standalone (paste URL)

1. Copy an Instagram video URL from anywhere (browser, messages, notes…).
2. Open the **Shortcuts** app and tap **Claude-InstaSaver**.
3. When prompted, paste the URL and tap **Done**.
4. The video is saved to your Camera Roll.

### Supported URL Formats

```
https://www.instagram.com/reel/ABC123xyz/
https://www.instagram.com/p/ABC123xyz/
https://www.instagram.com/tv/ABC123xyz/
```

---

## How It Works

The shortcut uses a two-method strategy to maximise compatibility:

### Method A — Instagram Internal API
```
GET https://www.instagram.com/p/{shortcode}/?__a=1&__d=dis
```
Instagram's internal JSON endpoint returns post metadata including `video_url` for public
video posts. This requires no authentication and works for most public accounts.

### Method B — Fallback (sssinstagram.com)
If Method A fails (private account visible to you, API change, network error), the shortcut
falls back to a third-party extraction service. The original URL is URL-encoded and sent to
the service's API; the response contains a direct download link.

### Download & Save
Once a direct MP4 URL is obtained, the shortcut:
1. Downloads the file using **Get Contents of URL** (native iOS networking).
2. Saves the binary to the **Camera Roll** via **Save to Photo Album**.
3. Displays a notification with the result.

---

## Troubleshooting

### "Untrusted Shortcut" — cannot add

Go to **Settings → Shortcuts → Allow Untrusted Shortcuts** and enable the toggle, then try importing again.

### "Invalid URL" error

Make sure you're sharing a full Instagram URL, not just a username or caption text.
The URL must contain `instagram.com` and a post path (`/p/`, `/reel/`, or `/tv/`).

### "Could Not Download Video" error

This happens when both extraction methods fail. Common causes:

| Cause | Fix |
|-------|-----|
| Private account (you don't follow) | Follow the account first; private posts require your session |
| Instagram API change | Check [ISSUES.md](ISSUES.md) for updated workarounds |
| Fallback service down | Try again in a few minutes |
| Post is a photo, not a video | Only video content can be saved |
| Rate limited by Instagram | Wait 10–15 minutes and try again |

### Video saves but won't play

The video file may have downloaded correctly but in an unsupported codec. Try playing it in
the **Files** app or **VLC** instead of Photos.

### Shortcut doesn't appear in Share Sheet

Open **Shortcuts** app → Long-press **Claude-InstaSaver** → **Details** → enable
**Show in Share Sheet**.

---

## Privacy

- The shortcut sends the Instagram URL to Instagram's own servers (Method A) and optionally
  to sssinstagram.com (Method B).
- **No data is collected or stored** by the shortcut itself.
- No account credentials are ever transmitted.
- You can review every network request in the **Shortcuts** app editor.

---

## Repository Structure

```
shortcuts/instagram-video-saver/
├── generate_shortcut.py      # Python script — generates the .shortcut binary
├── Claude-InstaSaver.shortcut # Ready-to-import iOS Shortcut (binary plist)
├── shortcut-source.json      # Human-readable JSON of the full workflow
├── README.md                 # This file
├── ISSUES.md                 # Known limitations & ToS notes
└── LICENSE                   # MIT License
```

---

## Contributing

Bug reports and improvements are welcome via GitHub Issues.

To modify the shortcut:
1. Edit `generate_shortcut.py` (the workflow is defined in the `ACTIONS` list).
2. Run `python3 generate_shortcut.py` to regenerate `Claude-InstaSaver.shortcut`.
3. Test on a real iOS device.
4. Submit a PR with both the updated script **and** the regenerated `.shortcut` file.

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| 1.0 | 2026-03 | Initial release — dual-method extraction, share sheet support, iOS 15+ |

---

## License

MIT — see [LICENSE](LICENSE).
