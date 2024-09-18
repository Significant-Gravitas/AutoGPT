/// `TestOption` is an enumeration of the available test options that can be selected in the skill tree view.
///
/// Each value of this enum represents a distinct test option that can be executed.
/// The `description` getter can be used to get the string representation of each test option.
enum TestOption {
  /// Represents the option to run a single test.
  runSingleTest,

  /// Represents the option to run a test suite including the selected node and its ancestors.
  runTestSuiteIncludingSelectedNodeAndAncestors,

  /// Represents the option to run all tests in a category.
  runAllTestsInCategory,
}

/// An extension on the `TestOption` enum to provide a string representation for each test option.
///
/// This extension adds a `description` getter on `TestOption` to easily retrieve the human-readable
/// string associated with each option. This is particularly helpful for UI display purposes.
extension TestOptionExtension on TestOption {
  /// Gets the string representation of the test option.
  ///
  /// Returns a human-readable string that describes the test option. This string is intended
  /// to be displayed in the UI for user selection.
  String get description {
    switch (this) {
      /// In case of a single test option, return the corresponding string.
      case TestOption.runSingleTest:
        return 'Run single test';

      /// In case of a test suite option that includes selected node and ancestors, return the corresponding string.
      case TestOption.runTestSuiteIncludingSelectedNodeAndAncestors:
        return 'Run test suite including selected node and ancestors';

      /// In case of an option to run all tests in a category, return the corresponding string.
      case TestOption.runAllTestsInCategory:
        return 'Run all tests in category';

      /// In case of an undefined or unknown test option, return a generic unknown string.
      /// This case should ideally never be hit if all enum values are handled.
      default:
        return 'Unknown';
    }
  }

  /// Converts a [description] string to its corresponding [TestOption] enum value.
  ///
  /// This method is helpful for converting string representations of test options
  /// received from various sources (like user input or server responses) into
  /// their type-safe enum equivalents.
  ///
  /// Returns the matching [TestOption] enum value if found, otherwise returns `null`.
  static TestOption? fromDescription(String description) {
    switch (description) {
      case 'Run single test':
        return TestOption.runSingleTest;
      case 'Run test suite including selected node and ancestors':
        return TestOption.runTestSuiteIncludingSelectedNodeAndAncestors;
      case 'Run all tests in category':
        return TestOption.runAllTestsInCategory;
      default:
        return null; // or throw an exception, or provide a default value, as per your requirement
    }
  }
}
