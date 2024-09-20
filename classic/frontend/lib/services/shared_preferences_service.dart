import 'package:shared_preferences/shared_preferences.dart';

class SharedPreferencesService {
  SharedPreferencesService._privateConstructor();

  static final SharedPreferencesService instance =
      SharedPreferencesService._privateConstructor();

  Future<SharedPreferences> _prefs = SharedPreferences.getInstance();

  /// Sets a boolean [value] for the given [key] in the shared preferences.
  ///
  /// Example:
  /// ```dart
  /// await prefsService.setBool('isLoggedIn', true);
  /// ```
  Future<void> setBool(String key, bool value) async {
    final prefs = await _prefs;
    prefs.setBool(key, value);
  }

  /// Sets a string [value] for the given [key] in the shared preferences.
  ///
  /// Example:
  /// ```dart
  /// await prefsService.setString('username', 'Alice');
  /// ```
  Future<void> setString(String key, String value) async {
    final prefs = await _prefs;
    prefs.setString(key, value);
  }

  /// Sets an integer [value] for the given [key] in the shared preferences.
  ///
  /// Example:
  /// ```dart
  /// await prefsService.setInt('age', 30);
  /// ```
  Future<void> setInt(String key, int value) async {
    final prefs = await _prefs;
    prefs.setInt(key, value);
  }

  /// Sets a list of strings [value] for the given [key] in the shared preferences.
  ///
  /// Example:
  /// ```dart
  /// await prefsService.setStringList('favorites', ['Apples', 'Bananas']);
  /// ```
  Future<void> setStringList(String key, List<String> value) async {
    final prefs = await _prefs;
    prefs.setStringList(key, value);
  }

  /// Retrieves a boolean value for the given [key] from the shared preferences.
  ///
  /// Returns `null` if the key does not exist.
  ///
  /// Example:
  /// ```dart
  /// bool? isLoggedIn = await prefsService.getBool('isLoggedIn');
  /// ```
  Future<bool?> getBool(String key) async {
    final prefs = await _prefs;
    return prefs.getBool(key);
  }

  /// Retrieves a string value for the given [key] from the shared preferences.
  ///
  /// Returns `null` if the key does not exist.
  ///
  /// Example:
  /// ```dart
  /// String? username = await prefsService.getString('username');
  /// ```
  Future<String?> getString(String key) async {
    final prefs = await _prefs;
    return prefs.getString(key);
  }

  /// Retrieves an integer value for the given [key] from the shared preferences.
  ///
  /// Returns `null` if the key does not exist.
  ///
  /// Example:
  /// ```dart
  /// int? age = await prefsService.getInt('age');
  /// ```
  Future<int?> getInt(String key) async {
    final prefs = await _prefs;
    return prefs.getInt(key);
  }

  /// Retrieves a list of strings for the given [key] from the shared preferences.
  ///
  /// Returns `null` if the key does not exist.
  ///
  /// Example:
  /// ```dart
  /// List<String>? favorites = await prefsService.getStringList('favorites');
  /// ```
  Future<List<String>?> getStringList(String key) async {
    final prefs = await _prefs;
    return prefs.getStringList(key);
  }
}
