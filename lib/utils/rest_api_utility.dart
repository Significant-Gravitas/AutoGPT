import 'dart:convert';
import 'package:http/http.dart' as http;

class RestApiUtility {
  String _baseUrl;

  RestApiUtility(this._baseUrl);

  void updateBaseURL(String newBaseURL) {
    _baseUrl = newBaseURL;
  }

  Future<Map<String, dynamic>> get(String endpoint) async {
    final response = await http.get(Uri.parse('$_baseUrl/$endpoint'));
    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to load data');
    }
  }

  Future<Map<String, dynamic>> post(
      String endpoint, Map<String, dynamic> payload) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/$endpoint'),
      body: json.encode(payload),
      headers: {"Content-Type": "application/json"},
    );
    if (response.statusCode == 200 || response.statusCode == 201) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to post data');
    }
  }
}
