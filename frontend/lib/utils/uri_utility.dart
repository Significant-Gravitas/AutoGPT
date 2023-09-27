class UriUtility {
  static bool isURL(String url) {
    // Validate if the URL string is empty, or contains spaces or invalid characters
    if (url.isEmpty || RegExp(r'[\s<>]').hasMatch(url)) {
      return false;
    }

    // Check for 'mailto:' at the start of the URL
    if (url.startsWith('mailto:')) {
      return false;
    }

    // Try to parse the URL, return false if parsing fails
    Uri? uri;
    try {
      uri = Uri.parse(url);
    } catch (e) {
      return false;
    }

    // Validate the URL has a scheme (protocol) and host
    if (uri.scheme.isEmpty || uri.host.isEmpty) {
      return false;
    }

    // Check if the URI has user info, which is not a common case for a valid HTTP/HTTPS URL
    if (uri.hasAuthority &&
        (uri.userInfo.isEmpty ||
            uri.userInfo.contains(':') && uri.userInfo.split(':').length > 2)) {
      return false;
    }

    // Validate the port number if exists
    if (uri.hasPort && (uri.port <= 0 || uri.port > 65535)) {
      return false;
    }

    return true;
  }
}
