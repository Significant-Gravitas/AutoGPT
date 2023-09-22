import 'package:flutter/material.dart';

/// [SettingsViewModel] is responsible for managing the state and logic
/// for the [SettingsView]. It extends [ChangeNotifier] to provide
/// reactive state management.
class SettingsViewModel extends ChangeNotifier {
  bool _isDarkModeEnabled = false; // State for Dark Mode
  bool _isDeveloperModeEnabled = false; // State for Developer Mode
  String _baseURL = ''; // State for Base URL
  int _continuousModeSteps = 1; // State for Continuous Mode Steps

  // Getters to access the private state variables
  bool get isDarkModeEnabled => _isDarkModeEnabled;
  bool get isDeveloperModeEnabled => _isDeveloperModeEnabled;
  String get baseURL => _baseURL;
  int get continuousModeSteps => _continuousModeSteps;

  /// Toggles the state of Dark Mode and notifies listeners.
  void toggleDarkMode(bool value) {
    _isDarkModeEnabled = value;
    notifyListeners();
    // TODO: Save to local storage or sync with the server
  }

  /// Toggles the state of Developer Mode and notifies listeners.
  void toggleDeveloperMode(bool value) {
    _isDeveloperModeEnabled = value;
    notifyListeners();
    // TODO: Save to local storage or sync with the server
  }

  /// Updates the state of Base URL and notifies listeners.
  void updateBaseURL(String value) {
    _baseURL = value;
    notifyListeners();
    // TODO: Save to local storage or sync with the server
  }

  /// Increments the number of Continuous Mode Steps and notifies listeners.
  void incrementContinuousModeSteps() {
    _continuousModeSteps += 1;
    notifyListeners();
    // TODO: Save to local storage or sync with the server
  }

  /// Decrements the number of Continuous Mode Steps and notifies listeners.
  void decrementContinuousModeSteps() {
    if (_continuousModeSteps > 1) {
      // Ensure that the number of steps is at least 1
      _continuousModeSteps -= 1;
      notifyListeners();
      // TODO: Save to local storage or sync with the server
    }
  }
}
