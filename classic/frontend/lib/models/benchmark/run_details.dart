// TODO: Remove the ability to have null values when benchmark implementation is complete
/// `RunDetails` encapsulates specific details about a benchmark run.
///
/// This class holds attributes such as the unique run identifier, the command used to initiate the run,
/// the time of completion, the time when the benchmark started, and the name of the test.
class RunDetails {
  /// The unique identifier for the benchmark run, typically a UUID.
  String runId;

  /// The command used to initiate the benchmark run.
  final String command;

  /// The completion time of the benchmark run as a `DateTime` object.
  final DateTime completionTime;

  /// The start time of the benchmark run as a `DateTime` object.
  final DateTime benchmarkStartTime;

  /// The name of the test associated with this benchmark run.
  final String testName;

  /// Constructs a new `RunDetails` instance.
  ///
  /// [runId]: The unique identifier for the benchmark run.
  /// [command]: The command used to initiate the run.
  /// [completionTime]: The completion time of the run.
  /// [benchmarkStartTime]: The start time of the run.
  /// [testName]: The name of the test.
  RunDetails({
    required this.runId,
    required this.command,
    required this.completionTime,
    required this.benchmarkStartTime,
    required this.testName,
  });

  /// Creates a `RunDetails` instance from a map.
  ///
  /// [json]: A map containing key-value pairs corresponding to `RunDetails` fields.
  ///
  /// Returns a new `RunDetails` populated with values from the map.
  factory RunDetails.fromJson(Map<String, dynamic> json) => RunDetails(
        runId: json['run_id'] ?? 'placerholder',
        command: json['command'] ?? 'placeholder',
        completionTime: json['completion_time'] == null
            ? DateTime.now()
            : DateTime.parse(json['completion_time']),
        benchmarkStartTime: json['benchmark_start_time'] == null
            ? DateTime.now()
            : DateTime.parse(json['benchmark_start_time']),
        testName: json['test_name'] ?? 'placeholder',
      );

  /// Converts the `RunDetails` instance to a map.
  ///
  /// Returns a map containing key-value pairs corresponding to `RunDetails` fields.
  Map<String, dynamic> toJson() => {
        'run_id': runId,
        'command': command,
        'completion_time': completionTime.toIso8601String(),
        'benchmark_start_time': benchmarkStartTime.toIso8601String(),
        'test_name': testName,
      };
}
