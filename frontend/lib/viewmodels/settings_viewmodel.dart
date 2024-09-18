import 'package:auto_gpt_flutter_client/services/auth_service.dart';
import 'package:auto_gpt_flutter_client/services/shared_preferences_service.dart';
import 'package:auto_gpt_flutter_client/utils/rest_api_utility.dart';
import 'package:flutter/material.dart';

/// [SettingsViewModel] is responsible for managing the state and logic
/// for the [SettingsView]. It extends [ChangeNotifier] to provide
/// reactive state management.
class SettingsViewModel extends ChangeNotifier {
  bool _isDarkModeEnabled = false; // State for Dark Mode
  bool _isDeveloperModeEnabled = false; // State for Developer Mode
  String _baseURL = ''; // State for Base URL
  int _continuousModeSteps = 1; // State for Continuous Mode Steps

  final RestApiUtility _restApiUtility;
  final SharedPreferencesService _prefsService;

  // Getters to access the private state variables
  bool get isDarkModeEnabled => _isDarkModeEnabled;
  bool get isDeveloperModeEnabled => _isDeveloperModeEnabled;
  String get baseURL => _baseURL;
  int get continuousModeSteps => _continuousModeSteps;

  final AuthService _authService = AuthService();

  SettingsViewModel(this._restApiUtility, this._prefsService) {
    _loadPreferences();
  }

  // Method to load stored preferences
  Future<void> _loadPreferences() async {
    _isDarkModeEnabled =
        await _prefsService.getBool('isDarkModeEnabled') ?? false;
    _isDeveloperModeEnabled =
        await _prefsService.getBool('isDeveloperModeEnabled') ?? true;
    _baseURL = await _prefsService.getString('baseURL') ??
        'http://127.0.0.1:8000/ap/v1';
    _restApiUtility.updateBaseURL(_baseURL);
    _continuousModeSteps =
        await _prefsService.getInt('continuousModeSteps') ?? 10;
    notifyListeners();
  }

  /// Toggles the state of Dark Mode and notifies listeners.
  Future<void> toggleDarkMode(bool value) async {
    _isDarkModeEnabled = value;
    notifyListeners();
    await _prefsService.setBool('isDarkModeEnabled', value);
  }

  /// Toggles the state of Developer Mode and notifies listeners.
  Future<void> toggleDeveloperMode(bool value) async {
    _isDeveloperModeEnabled = value;
    notifyListeners();
    await _prefsService.setBool('isDeveloperModeEnabled', value);
  }

  /// Updates the state of Base URL, notifies listeners, and updates the RestApiUtility baseURL.
  Future<void> updateBaseURL(String value) async {
    _baseURL = value;
    notifyListeners();
    await _prefsService.setString('baseURL', value);
    _restApiUtility.updateBaseURL(value);
  }

  /// Increments the number of Continuous Mode Steps and notifies listeners.
  Future<void> incrementContinuousModeSteps() async {
    _continuousModeSteps += 1;
    notifyListeners();
    await _prefsService.setInt('continuousModeSteps', _continuousModeSteps);
  }

  /// Decrements the number of Continuous Mode Steps and notifies listeners.
  Future<void> decrementContinuousModeSteps() async {
    if (_continuousModeSteps > 1) {
      // Ensure that the number of steps is at least 1
      _continuousModeSteps -= 1;
      notifyListeners();
      await _prefsService.setInt('continuousModeSteps', _continuousModeSteps);
    }
  }

  // Method to sign out
  Future<void> signOut() async {
    await _authService.signOut();
  }
}
