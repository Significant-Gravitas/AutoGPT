import 'package:http/http.dart' as http;
import 'dart:convert';

class UriUtility {
  static bool isURL(String url) {
    // Validate if the URL string is empty, or contains spaces or invalid characters
    if (url.isEmpty || RegExp(r'[\s<>]').hasMatch(url)) {
      print('URL is either empty or contains spaces/invalid characters.');
      return false;
    }

    // Check for 'mailto:' at the start of the URL
    if (url.startsWith('mailto:')) {
      print('URL starts with "mailto:".');
      return false;
    }

    // Try to parse the URL, return false if parsing fails
    Uri? uri;
    try {
      uri = Uri.parse(url);
    } catch (e) {
      print('URL parsing failed: $e');
      return false;
    }

    // Validate the URL has a scheme (protocol) and host
    if (uri.scheme.isEmpty || uri.host.isEmpty) {
      print('URL is missing a scheme (protocol) or host.');
      return false;
    }

    // Check if the URI has user info, which is not a common case for a valid HTTP/HTTPS URL
    if (uri.hasAuthority &&
        uri.userInfo.contains(':') &&
        uri.userInfo.split(':').length > 2) {
      print('URL contains invalid user info.');
      return false;
    }

    // Validate the port number if exists
    if (uri.hasPort && (uri.port <= 0 || uri.port > 65535)) {
      print('URL contains an invalid port number.');
      return false;
    }

    print('URL is valid.');
    return true;
  }

  Future<bool> isValidGitHubRepo(String repoUrl) async {
    var uri = Uri.parse(repoUrl);
    if (uri.host != 'github.com') {
      return false;
    }

    var segments = uri.pathSegments;
    if (segments.length < 2) {
      return false;
    }

    var user = segments[0];
    var repo = segments[1];

    var apiUri = Uri.https('api.github.com', '/repos/$user/$repo');

    var response = await http.get(apiUri);
    if (response.statusCode != 200) {
      return false;
    }

    var data = json.decode(response.body);
    return data is Map && data['full_name'] == '$user/$repo';
  }
}
