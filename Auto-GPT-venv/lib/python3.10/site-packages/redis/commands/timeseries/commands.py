from typing import Dict, List, Optional, Tuple, Union

from redis.exceptions import DataError
from redis.typing import KeyT, Number

ADD_CMD = "TS.ADD"
ALTER_CMD = "TS.ALTER"
CREATERULE_CMD = "TS.CREATERULE"
CREATE_CMD = "TS.CREATE"
DECRBY_CMD = "TS.DECRBY"
DELETERULE_CMD = "TS.DELETERULE"
DEL_CMD = "TS.DEL"
GET_CMD = "TS.GET"
INCRBY_CMD = "TS.INCRBY"
INFO_CMD = "TS.INFO"
MADD_CMD = "TS.MADD"
MGET_CMD = "TS.MGET"
MRANGE_CMD = "TS.MRANGE"
MREVRANGE_CMD = "TS.MREVRANGE"
QUERYINDEX_CMD = "TS.QUERYINDEX"
RANGE_CMD = "TS.RANGE"
REVRANGE_CMD = "TS.REVRANGE"


class TimeSeriesCommands:
    """RedisTimeSeries Commands."""

    def create(
        self,
        key: KeyT,
        retention_msecs: Optional[int] = None,
        uncompressed: Optional[bool] = False,
        labels: Optional[Dict[str, str]] = None,
        chunk_size: Optional[int] = None,
        duplicate_policy: Optional[str] = None,
    ):
        """
        Create a new time-series.

        Args:

        key:
            time-series key
        retention_msecs:
            Maximum age for samples compared to highest reported timestamp (in milliseconds).
            If None or 0 is passed then  the series is not trimmed at all.
        uncompressed:
            Changes data storage from compressed (by default) to uncompressed
        labels:
            Set of label-value pairs that represent metadata labels of the key.
        chunk_size:
            Memory size, in bytes, allocated for each data chunk.
            Must be a multiple of 8 in the range [128 .. 1048576].
        duplicate_policy:
            Policy for handling multiple samples with identical timestamps.
            Can be one of:
            - 'block': an error will occur for any out of order sample.
            - 'first': ignore the new value.
            - 'last': override with latest value.
            - 'min': only override if the value is lower than the existing value.
            - 'max': only override if the value is higher than the existing value.

        For more information: https://redis.io/commands/ts.create/
        """  # noqa
        params = [key]
        self._append_retention(params, retention_msecs)
        self._append_uncompressed(params, uncompressed)
        self._append_chunk_size(params, chunk_size)
        self._append_duplicate_policy(params, CREATE_CMD, duplicate_policy)
        self._append_labels(params, labels)

        return self.execute_command(CREATE_CMD, *params)

    def alter(
        self,
        key: KeyT,
        retention_msecs: Optional[int] = None,
        labels: Optional[Dict[str, str]] = None,
        chunk_size: Optional[int] = None,
        duplicate_policy: Optional[str] = None,
    ):
        """
        Update the retention, chunk size, duplicate policy, and labels of an existing
        time series.

        Args:

        key:
            time-series key
        retention_msecs:
            Maximum retention period, compared to maximal existing timestamp (in milliseconds).
            If None or 0 is passed then  the series is not trimmed at all.
        labels:
            Set of label-value pairs that represent metadata labels of the key.
        chunk_size:
            Memory size, in bytes, allocated for each data chunk.
            Must be a multiple of 8 in the range [128 .. 1048576].
        duplicate_policy:
            Policy for handling multiple samples with identical timestamps.
            Can be one of:
            - 'block': an error will occur for any out of order sample.
            - 'first': ignore the new value.
            - 'last': override with latest value.
            - 'min': only override if the value is lower than the existing value.
            - 'max': only override if the value is higher than the existing value.

        For more information: https://redis.io/commands/ts.alter/
        """  # noqa
        params = [key]
        self._append_retention(params, retention_msecs)
        self._append_chunk_size(params, chunk_size)
        self._append_duplicate_policy(params, ALTER_CMD, duplicate_policy)
        self._append_labels(params, labels)

        return self.execute_command(ALTER_CMD, *params)

    def add(
        self,
        key: KeyT,
        timestamp: Union[int, str],
        value: Number,
        retention_msecs: Optional[int] = None,
        uncompressed: Optional[bool] = False,
        labels: Optional[Dict[str, str]] = None,
        chunk_size: Optional[int] = None,
        duplicate_policy: Optional[str] = None,
    ):
        """
        Append (or create and append) a new sample to a time series.

        Args:

        key:
            time-series key
        timestamp:
            Timestamp of the sample. * can be used for automatic timestamp (using the system clock).
        value:
            Numeric data value of the sample
        retention_msecs:
            Maximum retention period, compared to maximal existing timestamp (in milliseconds).
            If None or 0 is passed then  the series is not trimmed at all.
        uncompressed:
            Changes data storage from compressed (by default) to uncompressed
        labels:
            Set of label-value pairs that represent metadata labels of the key.
        chunk_size:
            Memory size, in bytes, allocated for each data chunk.
            Must be a multiple of 8 in the range [128 .. 1048576].
        duplicate_policy:
            Policy for handling multiple samples with identical timestamps.
            Can be one of:
            - 'block': an error will occur for any out of order sample.
            - 'first': ignore the new value.
            - 'last': override with latest value.
            - 'min': only override if the value is lower than the existing value.
            - 'max': only override if the value is higher than the existing value.

        For more information: https://redis.io/commands/ts.add/
        """  # noqa
        params = [key, timestamp, value]
        self._append_retention(params, retention_msecs)
        self._append_uncompressed(params, uncompressed)
        self._append_chunk_size(params, chunk_size)
        self._append_duplicate_policy(params, ADD_CMD, duplicate_policy)
        self._append_labels(params, labels)

        return self.execute_command(ADD_CMD, *params)

    def madd(self, ktv_tuples: List[Tuple[KeyT, Union[int, str], Number]]):
        """
        Append (or create and append) a new `value` to series
        `key` with `timestamp`.
        Expects a list of `tuples` as (`key`,`timestamp`, `value`).
        Return value is an array with timestamps of insertions.

        For more information: https://redis.io/commands/ts.madd/
        """  # noqa
        params = []
        for ktv in ktv_tuples:
            params.extend(ktv)

        return self.execute_command(MADD_CMD, *params)

    def incrby(
        self,
        key: KeyT,
        value: Number,
        timestamp: Optional[Union[int, str]] = None,
        retention_msecs: Optional[int] = None,
        uncompressed: Optional[bool] = False,
        labels: Optional[Dict[str, str]] = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Increment (or create an time-series and increment) the latest sample's of a series.
        This command can be used as a counter or gauge that automatically gets history as a time series.

        Args:

        key:
            time-series key
        value:
            Numeric data value of the sample
        timestamp:
            Timestamp of the sample. * can be used for automatic timestamp (using the system clock).
        retention_msecs:
            Maximum age for samples compared to last event time (in milliseconds).
            If None or 0 is passed then  the series is not trimmed at all.
        uncompressed:
            Changes data storage from compressed (by default) to uncompressed
        labels:
            Set of label-value pairs that represent metadata labels of the key.
        chunk_size:
            Memory size, in bytes, allocated for each data chunk.

        For more information: https://redis.io/commands/ts.incrby/
        """  # noqa
        params = [key, value]
        self._append_timestamp(params, timestamp)
        self._append_retention(params, retention_msecs)
        self._append_uncompressed(params, uncompressed)
        self._append_chunk_size(params, chunk_size)
        self._append_labels(params, labels)

        return self.execute_command(INCRBY_CMD, *params)

    def decrby(
        self,
        key: KeyT,
        value: Number,
        timestamp: Optional[Union[int, str]] = None,
        retention_msecs: Optional[int] = None,
        uncompressed: Optional[bool] = False,
        labels: Optional[Dict[str, str]] = None,
        chunk_size: Optional[int] = None,
    ):
        """
        Decrement (or create an time-series and decrement) the latest sample's of a series.
        This command can be used as a counter or gauge that automatically gets history as a time series.

        Args:

        key:
            time-series key
        value:
            Numeric data value of the sample
        timestamp:
            Timestamp of the sample. * can be used for automatic timestamp (using the system clock).
        retention_msecs:
            Maximum age for samples compared to last event time (in milliseconds).
            If None or 0 is passed then  the series is not trimmed at all.
        uncompressed:
            Changes data storage from compressed (by default) to uncompressed
        labels:
            Set of label-value pairs that represent metadata labels of the key.
        chunk_size:
            Memory size, in bytes, allocated for each data chunk.

        For more information: https://redis.io/commands/ts.decrby/
        """  # noqa
        params = [key, value]
        self._append_timestamp(params, timestamp)
        self._append_retention(params, retention_msecs)
        self._append_uncompressed(params, uncompressed)
        self._append_chunk_size(params, chunk_size)
        self._append_labels(params, labels)

        return self.execute_command(DECRBY_CMD, *params)

    def delete(self, key: KeyT, from_time: int, to_time: int):
        """
        Delete all samples between two timestamps for a given time series.

        Args:

        key:
            time-series key.
        from_time:
            Start timestamp for the range deletion.
        to_time:
            End timestamp for the range deletion.

        For more information: https://redis.io/commands/ts.del/
        """  # noqa
        return self.execute_command(DEL_CMD, key, from_time, to_time)

    def createrule(
        self,
        source_key: KeyT,
        dest_key: KeyT,
        aggregation_type: str,
        bucket_size_msec: int,
        align_timestamp: Optional[int] = None,
    ):
        """
        Create a compaction rule from values added to `source_key` into `dest_key`.

        Args:

        source_key:
            Key name for source time series
        dest_key:
            Key name for destination (compacted) time series
        aggregation_type:
            Aggregation type: One of the following:
            [`avg`, `sum`, `min`, `max`, `range`, `count`, `first`, `last`, `std.p`,
            `std.s`, `var.p`, `var.s`, `twa`]
        bucket_size_msec:
            Duration of each bucket, in milliseconds
        align_timestamp:
            Assure that there is a bucket that starts at exactly align_timestamp and
            align all other buckets accordingly.

        For more information: https://redis.io/commands/ts.createrule/
        """  # noqa
        params = [source_key, dest_key]
        self._append_aggregation(params, aggregation_type, bucket_size_msec)
        if align_timestamp is not None:
            params.append(align_timestamp)

        return self.execute_command(CREATERULE_CMD, *params)

    def deleterule(self, source_key: KeyT, dest_key: KeyT):
        """
        Delete a compaction rule from `source_key` to `dest_key`..

        For more information: https://redis.io/commands/ts.deleterule/
        """  # noqa
        return self.execute_command(DELETERULE_CMD, source_key, dest_key)

    def __range_params(
        self,
        key: KeyT,
        from_time: Union[int, str],
        to_time: Union[int, str],
        count: Optional[int],
        aggregation_type: Optional[str],
        bucket_size_msec: Optional[int],
        filter_by_ts: Optional[List[int]],
        filter_by_min_value: Optional[int],
        filter_by_max_value: Optional[int],
        align: Optional[Union[int, str]],
        latest: Optional[bool],
        bucket_timestamp: Optional[str],
        empty: Optional[bool],
    ):
        """Create TS.RANGE and TS.REVRANGE arguments."""
        params = [key, from_time, to_time]
        self._append_latest(params, latest)
        self._append_filer_by_ts(params, filter_by_ts)
        self._append_filer_by_value(params, filter_by_min_value, filter_by_max_value)
        self._append_count(params, count)
        self._append_align(params, align)
        self._append_aggregation(params, aggregation_type, bucket_size_msec)
        self._append_bucket_timestamp(params, bucket_timestamp)
        self._append_empty(params, empty)

        return params

    def range(
        self,
        key: KeyT,
        from_time: Union[int, str],
        to_time: Union[int, str],
        count: Optional[int] = None,
        aggregation_type: Optional[str] = None,
        bucket_size_msec: Optional[int] = 0,
        filter_by_ts: Optional[List[int]] = None,
        filter_by_min_value: Optional[int] = None,
        filter_by_max_value: Optional[int] = None,
        align: Optional[Union[int, str]] = None,
        latest: Optional[bool] = False,
        bucket_timestamp: Optional[str] = None,
        empty: Optional[bool] = False,
    ):
        """
        Query a range in forward direction for a specific time-serie.

        Args:

        key:
            Key name for timeseries.
        from_time:
            Start timestamp for the range query. - can be used to express the minimum possible timestamp (0).
        to_time:
            End timestamp for range query, + can be used to express the maximum possible timestamp.
        count:
            Limits the number of returned samples.
        aggregation_type:
            Optional aggregation type. Can be one of [`avg`, `sum`, `min`, `max`,
            `range`, `count`, `first`, `last`, `std.p`, `std.s`, `var.p`, `var.s`, `twa`]
        bucket_size_msec:
            Time bucket for aggregation in milliseconds.
        filter_by_ts:
            List of timestamps to filter the result by specific timestamps.
        filter_by_min_value:
            Filter result by minimum value (must mention also filter by_max_value).
        filter_by_max_value:
            Filter result by maximum value (must mention also filter by_min_value).
        align:
            Timestamp for alignment control for aggregation.
        latest:
            Used when a time series is a compaction, reports the compacted value of the
            latest possibly partial bucket
        bucket_timestamp:
            Controls how bucket timestamps are reported. Can be one of [`-`, `low`, `+`,
            `high`, `~`, `mid`].
        empty:
            Reports aggregations for empty buckets.

        For more information: https://redis.io/commands/ts.range/
        """  # noqa
        params = self.__range_params(
            key,
            from_time,
            to_time,
            count,
            aggregation_type,
            bucket_size_msec,
            filter_by_ts,
            filter_by_min_value,
            filter_by_max_value,
            align,
            latest,
            bucket_timestamp,
            empty,
        )
        return self.execute_command(RANGE_CMD, *params)

    def revrange(
        self,
        key: KeyT,
        from_time: Union[int, str],
        to_time: Union[int, str],
        count: Optional[int] = None,
        aggregation_type: Optional[str] = None,
        bucket_size_msec: Optional[int] = 0,
        filter_by_ts: Optional[List[int]] = None,
        filter_by_min_value: Optional[int] = None,
        filter_by_max_value: Optional[int] = None,
        align: Optional[Union[int, str]] = None,
        latest: Optional[bool] = False,
        bucket_timestamp: Optional[str] = None,
        empty: Optional[bool] = False,
    ):
        """
        Query a range in reverse direction for a specific time-series.

        **Note**: This command is only available since RedisTimeSeries >= v1.4

        Args:

        key:
            Key name for timeseries.
        from_time:
            Start timestamp for the range query. - can be used to express the minimum possible timestamp (0).
        to_time:
            End timestamp for range query, + can be used to express the maximum possible timestamp.
        count:
            Limits the number of returned samples.
        aggregation_type:
            Optional aggregation type. Can be one of [`avg`, `sum`, `min`, `max`,
            `range`, `count`, `first`, `last`, `std.p`, `std.s`, `var.p`, `var.s`, `twa`]
        bucket_size_msec:
            Time bucket for aggregation in milliseconds.
        filter_by_ts:
            List of timestamps to filter the result by specific timestamps.
        filter_by_min_value:
            Filter result by minimum value (must mention also filter_by_max_value).
        filter_by_max_value:
            Filter result by maximum value (must mention also filter_by_min_value).
        align:
            Timestamp for alignment control for aggregation.
        latest:
            Used when a time series is a compaction, reports the compacted value of the
            latest possibly partial bucket
        bucket_timestamp:
            Controls how bucket timestamps are reported. Can be one of [`-`, `low`, `+`,
            `high`, `~`, `mid`].
        empty:
            Reports aggregations for empty buckets.

        For more information: https://redis.io/commands/ts.revrange/
        """  # noqa
        params = self.__range_params(
            key,
            from_time,
            to_time,
            count,
            aggregation_type,
            bucket_size_msec,
            filter_by_ts,
            filter_by_min_value,
            filter_by_max_value,
            align,
            latest,
            bucket_timestamp,
            empty,
        )
        return self.execute_command(REVRANGE_CMD, *params)

    def __mrange_params(
        self,
        aggregation_type: Optional[str],
        bucket_size_msec: Optional[int],
        count: Optional[int],
        filters: List[str],
        from_time: Union[int, str],
        to_time: Union[int, str],
        with_labels: Optional[bool],
        filter_by_ts: Optional[List[int]],
        filter_by_min_value: Optional[int],
        filter_by_max_value: Optional[int],
        groupby: Optional[str],
        reduce: Optional[str],
        select_labels: Optional[List[str]],
        align: Optional[Union[int, str]],
        latest: Optional[bool],
        bucket_timestamp: Optional[str],
        empty: Optional[bool],
    ):
        """Create TS.MRANGE and TS.MREVRANGE arguments."""
        params = [from_time, to_time]
        self._append_latest(params, latest)
        self._append_filer_by_ts(params, filter_by_ts)
        self._append_filer_by_value(params, filter_by_min_value, filter_by_max_value)
        self._append_with_labels(params, with_labels, select_labels)
        self._append_count(params, count)
        self._append_align(params, align)
        self._append_aggregation(params, aggregation_type, bucket_size_msec)
        self._append_bucket_timestamp(params, bucket_timestamp)
        self._append_empty(params, empty)
        params.extend(["FILTER"])
        params += filters
        self._append_groupby_reduce(params, groupby, reduce)
        return params

    def mrange(
        self,
        from_time: Union[int, str],
        to_time: Union[int, str],
        filters: List[str],
        count: Optional[int] = None,
        aggregation_type: Optional[str] = None,
        bucket_size_msec: Optional[int] = 0,
        with_labels: Optional[bool] = False,
        filter_by_ts: Optional[List[int]] = None,
        filter_by_min_value: Optional[int] = None,
        filter_by_max_value: Optional[int] = None,
        groupby: Optional[str] = None,
        reduce: Optional[str] = None,
        select_labels: Optional[List[str]] = None,
        align: Optional[Union[int, str]] = None,
        latest: Optional[bool] = False,
        bucket_timestamp: Optional[str] = None,
        empty: Optional[bool] = False,
    ):
        """
        Query a range across multiple time-series by filters in forward direction.

        Args:

        from_time:
            Start timestamp for the range query. `-` can be used to express the minimum possible timestamp (0).
        to_time:
            End timestamp for range query, `+` can be used to express the maximum possible timestamp.
        filters:
            filter to match the time-series labels.
        count:
            Limits the number of returned samples.
        aggregation_type:
            Optional aggregation type. Can be one of [`avg`, `sum`, `min`, `max`,
            `range`, `count`, `first`, `last`, `std.p`, `std.s`, `var.p`, `var.s`, `twa`]
        bucket_size_msec:
            Time bucket for aggregation in milliseconds.
        with_labels:
            Include in the reply all label-value pairs representing metadata labels of the time series.
        filter_by_ts:
            List of timestamps to filter the result by specific timestamps.
        filter_by_min_value:
            Filter result by minimum value (must mention also filter_by_max_value).
        filter_by_max_value:
            Filter result by maximum value (must mention also filter_by_min_value).
        groupby:
            Grouping by fields the results (must mention also reduce).
        reduce:
            Applying reducer functions on each group. Can be one of [`avg` `sum`, `min`,
            `max`, `range`, `count`, `std.p`, `std.s`, `var.p`, `var.s`].
        select_labels:
            Include in the reply only a subset of the key-value pair labels of a series.
        align:
            Timestamp for alignment control for aggregation.
        latest:
            Used when a time series is a compaction, reports the compacted
            value of the latest possibly partial bucket
        bucket_timestamp:
            Controls how bucket timestamps are reported. Can be one of [`-`, `low`, `+`,
            `high`, `~`, `mid`].
        empty:
            Reports aggregations for empty buckets.

        For more information: https://redis.io/commands/ts.mrange/
        """  # noqa
        params = self.__mrange_params(
            aggregation_type,
            bucket_size_msec,
            count,
            filters,
            from_time,
            to_time,
            with_labels,
            filter_by_ts,
            filter_by_min_value,
            filter_by_max_value,
            groupby,
            reduce,
            select_labels,
            align,
            latest,
            bucket_timestamp,
            empty,
        )

        return self.execute_command(MRANGE_CMD, *params)

    def mrevrange(
        self,
        from_time: Union[int, str],
        to_time: Union[int, str],
        filters: List[str],
        count: Optional[int] = None,
        aggregation_type: Optional[str] = None,
        bucket_size_msec: Optional[int] = 0,
        with_labels: Optional[bool] = False,
        filter_by_ts: Optional[List[int]] = None,
        filter_by_min_value: Optional[int] = None,
        filter_by_max_value: Optional[int] = None,
        groupby: Optional[str] = None,
        reduce: Optional[str] = None,
        select_labels: Optional[List[str]] = None,
        align: Optional[Union[int, str]] = None,
        latest: Optional[bool] = False,
        bucket_timestamp: Optional[str] = None,
        empty: Optional[bool] = False,
    ):
        """
        Query a range across multiple time-series by filters in reverse direction.

        Args:

        from_time:
            Start timestamp for the range query. - can be used to express the minimum possible timestamp (0).
        to_time:
            End timestamp for range query, + can be used to express the maximum possible timestamp.
        filters:
            Filter to match the time-series labels.
        count:
            Limits the number of returned samples.
        aggregation_type:
            Optional aggregation type. Can be one of [`avg`, `sum`, `min`, `max`,
            `range`, `count`, `first`, `last`, `std.p`, `std.s`, `var.p`, `var.s`, `twa`]
        bucket_size_msec:
            Time bucket for aggregation in milliseconds.
        with_labels:
            Include in the reply all label-value pairs representing metadata labels of the time series.
        filter_by_ts:
            List of timestamps to filter the result by specific timestamps.
        filter_by_min_value:
            Filter result by minimum value (must mention also filter_by_max_value).
        filter_by_max_value:
            Filter result by maximum value (must mention also filter_by_min_value).
        groupby:
            Grouping by fields the results (must mention also reduce).
        reduce:
            Applying reducer functions on each group. Can be one of [`avg` `sum`, `min`,
            `max`, `range`, `count`, `std.p`, `std.s`, `var.p`, `var.s`].
        select_labels:
            Include in the reply only a subset of the key-value pair labels of a series.
        align:
            Timestamp for alignment control for aggregation.
        latest:
            Used when a time series is a compaction, reports the compacted
            value of the latest possibly partial bucket
        bucket_timestamp:
            Controls how bucket timestamps are reported. Can be one of [`-`, `low`, `+`,
            `high`, `~`, `mid`].
        empty:
            Reports aggregations for empty buckets.

        For more information: https://redis.io/commands/ts.mrevrange/
        """  # noqa
        params = self.__mrange_params(
            aggregation_type,
            bucket_size_msec,
            count,
            filters,
            from_time,
            to_time,
            with_labels,
            filter_by_ts,
            filter_by_min_value,
            filter_by_max_value,
            groupby,
            reduce,
            select_labels,
            align,
            latest,
            bucket_timestamp,
            empty,
        )

        return self.execute_command(MREVRANGE_CMD, *params)

    def get(self, key: KeyT, latest: Optional[bool] = False):
        """# noqa
        Get the last sample of `key`.
        `latest` used when a time series is a compaction, reports the compacted
        value of the latest (possibly partial) bucket

        For more information: https://redis.io/commands/ts.get/
        """  # noqa
        params = [key]
        self._append_latest(params, latest)
        return self.execute_command(GET_CMD, *params)

    def mget(
        self,
        filters: List[str],
        with_labels: Optional[bool] = False,
        select_labels: Optional[List[str]] = None,
        latest: Optional[bool] = False,
    ):
        """# noqa
        Get the last samples matching the specific `filter`.

        Args:

        filters:
            Filter to match the time-series labels.
        with_labels:
            Include in the reply all label-value pairs representing metadata
            labels of the time series.
        select_labels:
            Include in the reply only a subset of the key-value pair labels of a series.
        latest:
            Used when a time series is a compaction, reports the compacted
            value of the latest possibly partial bucket

        For more information: https://redis.io/commands/ts.mget/
        """  # noqa
        params = []
        self._append_latest(params, latest)
        self._append_with_labels(params, with_labels, select_labels)
        params.extend(["FILTER"])
        params += filters
        return self.execute_command(MGET_CMD, *params)

    def info(self, key: KeyT):
        """# noqa
        Get information of `key`.

        For more information: https://redis.io/commands/ts.info/
        """  # noqa
        return self.execute_command(INFO_CMD, key)

    def queryindex(self, filters: List[str]):
        """# noqa
        Get all time series keys matching the `filter` list.

        For more information: https://redis.io/commands/ts.queryindex/
        """  # noq
        return self.execute_command(QUERYINDEX_CMD, *filters)

    @staticmethod
    def _append_uncompressed(params: List[str], uncompressed: Optional[bool]):
        """Append UNCOMPRESSED tag to params."""
        if uncompressed:
            params.extend(["UNCOMPRESSED"])

    @staticmethod
    def _append_with_labels(
        params: List[str],
        with_labels: Optional[bool],
        select_labels: Optional[List[str]],
    ):
        """Append labels behavior to params."""
        if with_labels and select_labels:
            raise DataError(
                "with_labels and select_labels cannot be provided together."
            )

        if with_labels:
            params.extend(["WITHLABELS"])
        if select_labels:
            params.extend(["SELECTED_LABELS", *select_labels])

    @staticmethod
    def _append_groupby_reduce(
        params: List[str], groupby: Optional[str], reduce: Optional[str]
    ):
        """Append GROUPBY REDUCE property to params."""
        if groupby is not None and reduce is not None:
            params.extend(["GROUPBY", groupby, "REDUCE", reduce.upper()])

    @staticmethod
    def _append_retention(params: List[str], retention: Optional[int]):
        """Append RETENTION property to params."""
        if retention is not None:
            params.extend(["RETENTION", retention])

    @staticmethod
    def _append_labels(params: List[str], labels: Optional[List[str]]):
        """Append LABELS property to params."""
        if labels:
            params.append("LABELS")
            for k, v in labels.items():
                params.extend([k, v])

    @staticmethod
    def _append_count(params: List[str], count: Optional[int]):
        """Append COUNT property to params."""
        if count is not None:
            params.extend(["COUNT", count])

    @staticmethod
    def _append_timestamp(params: List[str], timestamp: Optional[int]):
        """Append TIMESTAMP property to params."""
        if timestamp is not None:
            params.extend(["TIMESTAMP", timestamp])

    @staticmethod
    def _append_align(params: List[str], align: Optional[Union[int, str]]):
        """Append ALIGN property to params."""
        if align is not None:
            params.extend(["ALIGN", align])

    @staticmethod
    def _append_aggregation(
        params: List[str],
        aggregation_type: Optional[str],
        bucket_size_msec: Optional[int],
    ):
        """Append AGGREGATION property to params."""
        if aggregation_type is not None:
            params.extend(["AGGREGATION", aggregation_type, bucket_size_msec])

    @staticmethod
    def _append_chunk_size(params: List[str], chunk_size: Optional[int]):
        """Append CHUNK_SIZE property to params."""
        if chunk_size is not None:
            params.extend(["CHUNK_SIZE", chunk_size])

    @staticmethod
    def _append_duplicate_policy(
        params: List[str], command: Optional[str], duplicate_policy: Optional[str]
    ):
        """Append DUPLICATE_POLICY property to params on CREATE
        and ON_DUPLICATE on ADD.
        """
        if duplicate_policy is not None:
            if command == "TS.ADD":
                params.extend(["ON_DUPLICATE", duplicate_policy])
            else:
                params.extend(["DUPLICATE_POLICY", duplicate_policy])

    @staticmethod
    def _append_filer_by_ts(params: List[str], ts_list: Optional[List[int]]):
        """Append FILTER_BY_TS property to params."""
        if ts_list is not None:
            params.extend(["FILTER_BY_TS", *ts_list])

    @staticmethod
    def _append_filer_by_value(
        params: List[str], min_value: Optional[int], max_value: Optional[int]
    ):
        """Append FILTER_BY_VALUE property to params."""
        if min_value is not None and max_value is not None:
            params.extend(["FILTER_BY_VALUE", min_value, max_value])

    @staticmethod
    def _append_latest(params: List[str], latest: Optional[bool]):
        """Append LATEST property to params."""
        if latest:
            params.append("LATEST")

    @staticmethod
    def _append_bucket_timestamp(params: List[str], bucket_timestamp: Optional[str]):
        """Append BUCKET_TIMESTAMP property to params."""
        if bucket_timestamp is not None:
            params.extend(["BUCKETTIMESTAMP", bucket_timestamp])

    @staticmethod
    def _append_empty(params: List[str], empty: Optional[bool]):
        """Append EMPTY property to params."""
        if empty:
            params.append("EMPTY")
