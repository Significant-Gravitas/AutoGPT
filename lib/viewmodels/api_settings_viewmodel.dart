import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ApiSettingsViewModel with ChangeNotifier {
  String _baseURL = "http://127.0.0.1:8000";
  SharedPreferences? _prefs;

  ApiSettingsViewModel() {
    _loadBaseURL();
  }

  String get baseURL => _baseURL;

  void _loadBaseURL() async {
    _prefs = await SharedPreferences.getInstance();
    _baseURL = _prefs?.getString('baseURL') ?? _baseURL;
    notifyListeners();
  }

  void updateBaseURL(String newURL) async {
    _baseURL = newURL;
    _prefs ??= await SharedPreferences.getInstance();
    _prefs?.setString('baseURL', newURL);
    notifyListeners();
  }
}
