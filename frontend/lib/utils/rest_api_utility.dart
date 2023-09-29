import 'dart:convert';
import 'dart:typed_data';
import 'package:auto_gpt_flutter_client/models/benchmark/api_type.dart';
import 'package:http/http.dart' as http;

class RestApiUtility {
  String _agentBaseUrl;
  final String _benchmarkBaseUrl = "http://127.0.0.1:8080/ap/v1";
  final String _leaderboardBaseUrl = "https://leaderboard.agpt.co";

  RestApiUtility(this._agentBaseUrl);

  void updateBaseURL(String newBaseURL) {
    _agentBaseUrl = newBaseURL;
  }

  String _getEffectiveBaseUrl(ApiType apiType) {
    switch (apiType) {
      case ApiType.agent:
        return _agentBaseUrl;
      case ApiType.benchmark:
        return _benchmarkBaseUrl;
      case ApiType.leaderboard:
        return _leaderboardBaseUrl;
      default:
        return _agentBaseUrl;
    }
  }

  Future<Map<String, dynamic>> get(String endpoint,
      {ApiType apiType = ApiType.agent}) async {
    final effectiveBaseUrl = _getEffectiveBaseUrl(apiType);
    final response = await http.get(Uri.parse('$effectiveBaseUrl/$endpoint'));
    if (response.statusCode == 200) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to load data');
    }
  }

  Future<Map<String, dynamic>> post(
      String endpoint, Map<String, dynamic> payload,
      {ApiType apiType = ApiType.agent}) async {
    final effectiveBaseUrl = _getEffectiveBaseUrl(apiType);
    final response = await http.post(
      Uri.parse('$effectiveBaseUrl/$endpoint'),
      body: json.encode(payload),
      headers: {"Content-Type": "application/json"},
    );
    if (response.statusCode == 200 || response.statusCode == 201) {
      return json.decode(response.body);
    } else {
      // TODO: We are bubbling up the full response to show better errors on the UI.
      // Let's put some thought into how we would like to structure this.
      throw response;
    }
  }

  Future<Map<String, dynamic>> put(
      String endpoint, Map<String, dynamic> payload,
      {ApiType apiType = ApiType.agent}) async {
    final effectiveBaseUrl = _getEffectiveBaseUrl(apiType);
    final response = await http.put(
      Uri.parse('$effectiveBaseUrl/$endpoint'),
      body: json.encode(payload),
      headers: {"Content-Type": "application/json"},
    );
    if (response.statusCode == 200 || response.statusCode == 201) {
      return json.decode(response.body);
    } else {
      throw Exception('Failed to update data with PUT request');
    }
  }

  Future<Uint8List> getBinary(String endpoint,
      {ApiType apiType = ApiType.agent}) async {
    final effectiveBaseUrl = _getEffectiveBaseUrl(apiType);
    final response = await http.get(
      Uri.parse('$effectiveBaseUrl/$endpoint'),
      headers: {"Content-Type": "application/octet-stream"},
    );

    if (response.statusCode == 200) {
      return response.bodyBytes;
    } else if (response.statusCode == 404) {
      throw Exception('Resource not found');
    } else {
      throw Exception('Failed to load binary data');
    }
  }
}
