import 'package:auto_gpt_flutter_client/models/benchmark/config.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/metrics.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/repository_info.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/run_details.dart';
import 'package:auto_gpt_flutter_client/models/benchmark/task_info.dart';

// TODO: Remove the ability to have null values when benchmark implementation is complete
/// `BenchmarkRun` represents a complete benchmark run and encapsulates all associated data.
///
/// This class is a comprehensive model that includes various sub-models, each corresponding to a different aspect of a benchmark run.
/// It includes repository information, run details, task information, metrics, a flag indicating if the cutoff was reached, and configuration settings.
class BenchmarkRun {
  /// Information about the repository and team associated with the benchmark run.
  final RepositoryInfo repositoryInfo;

  /// Specific details about the benchmark run, like unique run identifier, command, and timings.
  final RunDetails runDetails;

  /// Information about the task being benchmarked, including its description and expected answer.
  final TaskInfo taskInfo;

  /// Performance metrics related to the benchmark run.
  final Metrics metrics;

  /// A boolean flag indicating whether the benchmark run reached a certain cutoff.
  final bool reachedCutoff;

  /// Configuration settings related to the benchmark run.
  final Config config;

  /// Constructs a new `BenchmarkRun` instance.
  ///
  /// [repositoryInfo]: Information about the repository and team.
  /// [runDetails]: Specific details about the benchmark run.
  /// [taskInfo]: Information about the task being benchmarked.
  /// [metrics]: Performance metrics for the benchmark run.
  /// [reachedCutoff]: A flag indicating if the benchmark run reached a certain cutoff.
  /// [config]: Configuration settings for the benchmark run.
  BenchmarkRun({
    required this.repositoryInfo,
    required this.runDetails,
    required this.taskInfo,
    required this.metrics,
    required this.reachedCutoff,
    required this.config,
  });

  /// Creates a `BenchmarkRun` instance from a map.
  ///
  /// [json]: A map containing key-value pairs corresponding to `BenchmarkRun` fields.
  ///
  /// Returns a new `BenchmarkRun` populated with values from the map.
  factory BenchmarkRun.fromJson(Map<String, dynamic> json) => BenchmarkRun(
        repositoryInfo: RepositoryInfo.fromJson(json['repository_info']),
        runDetails: RunDetails.fromJson(json['run_details']),
        taskInfo: TaskInfo.fromJson(json['task_info']),
        metrics: Metrics.fromJson(json['metrics']),
        reachedCutoff: json['reached_cutoff'] ?? false,
        config: Config.fromJson(json['config']),
      );

  /// Converts the `BenchmarkRun` instance to a map.
  ///
  /// Returns a map containing key-value pairs corresponding to `BenchmarkRun` fields.
  Map<String, dynamic> toJson() => {
        'repository_info': repositoryInfo.toJson(),
        'run_details': runDetails.toJson(),
        'task_info': taskInfo.toJson(),
        'metrics': metrics.toJson(),
        'reached_cutoff': reachedCutoff,
        'config': config.toJson(),
      };
}
