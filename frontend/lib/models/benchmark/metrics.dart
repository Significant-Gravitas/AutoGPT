// TODO: Remove the ability to have null values when benchmark implementation is complete
/// `Metrics` holds key performance metrics related to a benchmark test run.
///
/// The class encapsulates various data points like difficulty, success rate,
/// whether the task was attempted, the percentage of success, and other performance metrics.
class Metrics {
  /// The perceived difficulty level of the test, usually represented as a string.
  final String difficulty;

  /// A boolean indicating whether the test was successful.
  final bool success;

  /// A boolean indicating whether the test was attempted.
  final bool attempted;

  /// The percentage of success in the test, represented as a double.
  final double successPercentage;

  /// The cost metric, can be any type depending on what is being measured (time, resources, etc.).
  /// It is dynamic to allow for various types.
  final dynamic cost;

  /// The total runtime of the test, represented as a string.
  final String runTime;

  /// Constructs a new `Metrics` instance.
  ///
  /// [difficulty]: The perceived difficulty level of the test.
  /// [success]: A boolean indicating the success status of the test.
  /// [attempted]: A boolean indicating if the test was attempted.
  /// [successPercentage]: The success rate as a percentage.
  /// [cost]: The cost metric for the test.
  /// [runTime]: The total runtime of the test.
  Metrics({
    required this.difficulty,
    required this.success,
    required this.attempted,
    required this.successPercentage,
    required this.cost,
    required this.runTime,
  });

  /// Creates a `Metrics` instance from a map.
  ///
  /// [json]: A map containing key-value pairs corresponding to `Metrics` fields.
  ///
  /// Returns a new `Metrics` populated with values from the map.
  factory Metrics.fromJson(Map<String, dynamic> json) => Metrics(
        difficulty: json['difficulty'] ?? 'placeholder',
        success: json['success'],
        attempted: json['attempted'] ?? false,
        successPercentage: (json['success_percentage'] != null)
            ? json['success_percentage'].toDouble()
            : 0.0,
        cost: json['cost'] ?? 'placeholder',
        runTime: json['run_time'] ?? 'placeholder',
      );

  /// Converts the `Metrics` instance to a map.
  ///
  /// Returns a map containing key-value pairs corresponding to `Metrics` fields.
  Map<String, dynamic> toJson() => {
        'difficulty': difficulty,
        'success': success,
        'attempted': attempted,
        'success_percentage': successPercentage,
        'cost': cost,
        'run_time': runTime,
      };
}
