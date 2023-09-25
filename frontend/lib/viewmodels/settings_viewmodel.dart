import 'package:auto_gpt_flutter_client/services/auth_service.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// [SettingsViewModel] is responsible for managing the state and logic
/// for the [SettingsView]. It extends [ChangeNotifier] to provide
/// reactive state management.
class SettingsViewModel extends ChangeNotifier {
  bool _isDarkModeEnabled = false; // State for Dark Mode
  bool _isDeveloperModeEnabled = false; // State for Developer Mode
  String _baseURL = ''; // State for Base URL
  int _continuousModeSteps = 1; // State for Continuous Mode Steps

  final RestApiUtility _restApiUtility;

  // Getters to access the private state variables
  bool get isDarkModeEnabled => _isDarkModeEnabled;
  bool get isDeveloperModeEnabled => _isDeveloperModeEnabled;
  String get baseURL => _baseURL;
  int get continuousModeSteps => _continuousModeSteps;

  final AuthService _authService = AuthService();

  SettingsViewModel(this._restApiUtility) {
    _loadPreferences();
  }

  // Method to load stored preferences
  Future<void> _loadPreferences() async {
    final prefs = await SharedPreferences.getInstance();
    _isDarkModeEnabled = prefs.getBool('isDarkModeEnabled') ?? false;
    _isDeveloperModeEnabled = prefs.getBool('isDeveloperModeEnabled') ?? false;
    _baseURL = prefs.getString('baseURL') ?? 'http://127.0.0.1:8000/ap/v1';
    _restApiUtility.updateBaseURL(_baseURL);
    _continuousModeSteps = prefs.getInt('continuousModeSteps') ?? 10;
    notifyListeners();
  }

  /// Toggles the state of Dark Mode and notifies listeners.
  void toggleDarkMode(bool value) {
    _isDarkModeEnabled = value;
    notifyListeners();
    _saveBoolPreference('isDarkModeEnabled', value);
  }

  /// Toggles the state of Developer Mode and notifies listeners.
  void toggleDeveloperMode(bool value) {
    _isDeveloperModeEnabled = value;
    notifyListeners();
    _saveBoolPreference('isDeveloperModeEnabled', value);
  }

  /// Updates the state of Base URL, notifies listeners, and updates the RestApiUtility baseURL.
  void updateBaseURL(String value) {
    _baseURL = value;
    notifyListeners();
    _saveStringPreference('baseURL', value);
    _restApiUtility.updateBaseURL(value);
  }

  /// Increments the number of Continuous Mode Steps and notifies listeners.
  void incrementContinuousModeSteps() {
    _continuousModeSteps += 1;
    notifyListeners();
    _saveIntPreference('continuousModeSteps', _continuousModeSteps);
  }

  /// Decrements the number of Continuous Mode Steps and notifies listeners.
  void decrementContinuousModeSteps() {
    if (_continuousModeSteps > 1) {
      // Ensure that the number of steps is at least 1
      _continuousModeSteps -= 1;
      notifyListeners();
      _saveIntPreference('continuousModeSteps', _continuousModeSteps);
    }
  }

  // TODO: Create a service for interacting with shared preferences
  // Helper methods to save preferences
  Future<void> _saveBoolPreference(String key, bool value) async {
    final prefs = await SharedPreferences.getInstance();
    prefs.setBool(key, value);
  }

  Future<void> _saveStringPreference(String key, String value) async {
    final prefs = await SharedPreferences.getInstance();
    prefs.setString(key, value);
  }

  Future<void> _saveIntPreference(String key, int value) async {
    final prefs = await SharedPreferences.getInstance();
    prefs.setInt(key, value);
  }

  // Method to sign out
  Future<void> signOut() async {
    await _authService.signOut();
  }
}
