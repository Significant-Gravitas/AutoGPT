import Foundation

struct InstagramPost {
    let caption: String?
    let thumbnailURL: URL?
    let title: String?
}

enum InstagramScraperError: LocalizedError {
    case invalidURL
    case fetchFailed(String)
    case notInstagramURL

    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "The URL you entered is not valid."
        case .fetchFailed(let reason):
            return "Could not fetch the Instagram post: \(reason)"
        case .notInstagramURL:
            return "Please enter a valid Instagram post URL (instagram.com/p/...)."
        }
    }
}

actor InstagramScraperService {

    private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 15
        config.timeoutIntervalForResource = 30
        return URLSession(configuration: config)
    }()

    // iPhone User-Agent so Instagram returns the full HTML page
    private let userAgent = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) " +
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"

    func scrape(url rawURL: String) async throws -> InstagramPost {
        guard let url = URL(string: rawURL) else {
            throw InstagramScraperError.invalidURL
        }
        guard let host = url.host, host.contains("instagram.com") else {
            throw InstagramScraperError.notInstagramURL
        }

        // Tier 1: Instagram oEmbed – fast, returns thumbnail_url and a title snippet
        if let post = try? await fetchOEmbed(url: rawURL) {
            // oEmbed title is often short; augment with HTML caption if possible
            let htmlCaption = try? await fetchHTMLCaption(url: url)
            return InstagramPost(
                caption: htmlCaption ?? post.caption,
                thumbnailURL: post.thumbnailURL,
                title: post.title
            )
        }

        // Tier 2: Direct HTML fetch + Open Graph meta tag parsing
        if let post = try? await fetchHTMLPost(url: url) {
            return post
        }

        // Tier 3: Return empty post so caller can show partial data / prompt user
        return InstagramPost(caption: nil, thumbnailURL: nil, title: nil)
    }

    // MARK: - Private

    private func fetchOEmbed(url: String) async throws -> InstagramPost {
        let encoded = url.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? url
        guard let oembedURL = URL(string: "https://api.instagram.com/oembed/?url=\(encoded)&maxwidth=640&omitscript=true") else {
            throw InstagramScraperError.invalidURL
        }

        let (data, response) = try await session.data(from: oembedURL)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw InstagramScraperError.fetchFailed("oEmbed returned non-200")
        }

        let json = try JSONDecoder().decode(OEmbedResponse.self, from: data)
        let thumbnailURL = json.thumbnail_url.flatMap { URL(string: $0) }
        return InstagramPost(caption: json.title, thumbnailURL: thumbnailURL, title: json.title)
    }

    private func fetchHTMLCaption(url: URL) async throws -> String? {
        let html = try await fetchHTML(url: url)
        return extractMetaContent(from: html, property: "og:description")
    }

    private func fetchHTMLPost(url: URL) async throws -> InstagramPost {
        let html = try await fetchHTML(url: url)
        let caption = extractMetaContent(from: html, property: "og:description")
        let imageStr = extractMetaContent(from: html, property: "og:image")
        let thumbnailURL = imageStr.flatMap { URL(string: $0) }
        let title = extractMetaContent(from: html, property: "og:title")
        return InstagramPost(caption: caption, thumbnailURL: thumbnailURL, title: title)
    }

    private func fetchHTML(url: URL) async throws -> String {
        var request = URLRequest(url: url)
        request.setValue(userAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                         forHTTPHeaderField: "Accept")
        request.setValue("en-US,en;q=0.9", forHTTPHeaderField: "Accept-Language")

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            throw InstagramScraperError.fetchFailed("HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)")
        }
        return String(data: data, encoding: .utf8) ?? ""
    }

    /// Extracts content="…" from <meta property="og:xxx"> or <meta name="og:xxx">
    private func extractMetaContent(from html: String, property: String) -> String? {
        // Match both property= and name= variants
        let patterns = [
            "property=\"\(property)\"[^>]*content=\"([^\"]+)\"",
            "content=\"([^\"]+)\"[^>]*property=\"\(property)\"",
            "name=\"\(property)\"[^>]*content=\"([^\"]+)\"",
        ]
        for pattern in patterns {
            if let range = html.range(of: pattern, options: .regularExpression),
               let captureRange = extractFirstCapture(html: html, pattern: pattern) {
                _ = range
                return captureRange.htmlDecoded
            }
        }
        return nil
    }

    private func extractFirstCapture(html: String, pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else { return nil }
        let nsHtml = html as NSString
        guard let match = regex.firstMatch(in: html, range: NSRange(location: 0, length: nsHtml.length)),
              match.numberOfRanges > 1 else { return nil }
        let range = match.range(at: 1)
        guard range.location != NSNotFound else { return nil }
        return nsHtml.substring(with: range)
    }
}

// MARK: - Decodable helpers

private struct OEmbedResponse: Decodable {
    let title: String?
    let thumbnail_url: String?
}

private extension String {
    var htmlDecoded: String {
        let decoded = self
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&apos;", with: "'")
        return decoded
    }
}
