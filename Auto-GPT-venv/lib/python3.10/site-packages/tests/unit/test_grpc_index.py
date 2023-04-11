from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import pinecone
from pinecone import DescribeIndexStatsRequest
from pinecone.core.grpc.protos.vector_service_pb2 import Vector, DescribeIndexStatsRequest, UpdateRequest, \
    UpsertRequest, FetchRequest, QueryRequest, DeleteRequest, QueryVector, UpsertResponse, SparseValues
from pinecone.core.utils import dict_to_proto_struct


class TestGrpcIndex:

    def setup_method(self):
        self.vector_dim = 8
        self.vals1 = [0.1] * self.vector_dim
        self.vals2 = [0.2] * self.vector_dim
        self.sparse_indices_1 = [1, 8, 42]
        self.sparse_values_1 = [0.8, 0.9, 0.42]
        self.sparse_indices_2 = [1, 3, 5]
        self.sparse_values_2 = [0.7, 0.3, 0.31415]
        self.md1 = {'genre': 'action', 'year': 2021}
        self.md2 = {'genre': 'documentary', 'year': 2020}
        self.filter1 = {'genre': {'$in': ['action']}}
        self.filter2 = {'year': {'$eq': 2020}}

        pinecone.init(api_key='example-key')
        self.index = pinecone.GRPCIndex('example-name')

        self.expected_vec1 = Vector(id='vec1', values=self.vals1, metadata={})
        self.expected_vec2 = Vector(id='vec2', values=self.vals2, metadata={})
        self.expected_vec_md1 = Vector(id='vec1', values=self.vals1, metadata=dict_to_proto_struct(self.md1))
        self.expected_vec_md2 = Vector(id='vec2', values=self.vals2, metadata=dict_to_proto_struct(self.md2))
        self.expected_vec_md_sparse1 = Vector(id='vec1', values=self.vals1, metadata=dict_to_proto_struct(self.md1),
               sparse_values=SparseValues(indices=self.sparse_indices_1, values=self.sparse_values_1))
        self.expected_vec_md_sparse2 = Vector(id='vec2', values=self.vals2, metadata=dict_to_proto_struct(self.md2),
               sparse_values=SparseValues(indices=self.sparse_indices_2, values=self.sparse_values_2))


        # region: upsert tests

    def _assert_called_once(self, vectors, async_call=False):
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Upsert.future if async_call else self.index.stub.Upsert,
            UpsertRequest(
                vectors=vectors,
                namespace='ns'),
            timeout=None
        )

    def test_upsert_tuplesOfIdVec_UpserWithoutMD(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.upsert([('vec1', self.vals1), ('vec2', self.vals2)], namespace='ns')
        self._assert_called_once([
                    self.expected_vec1,
                    self.expected_vec2
        ])


    def test_upsert_tuplesOfIdVecMD_UpsertVectorsWithMD(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.upsert([('vec1', self.vals1, self.md1), ('vec2', self.vals2, self.md2)], namespace='ns')
        self._assert_called_once([
                    self.expected_vec_md1,
                    self.expected_vec_md2],
        )

    def test_upsert_vectors_upsertInputVectors(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.upsert([self.expected_vec_md1,
                           self.expected_vec_md2],
                          namespace='ns')
        self._assert_called_once([
                    self.expected_vec_md1,
                    self.expected_vec_md2],
        )


    def test_upsert_vectors_upsertInputVectorsSparse(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.upsert([Vector(id='vec1', values=self.vals1, metadata=dict_to_proto_struct(self.md1),
                                  sparse_values=SparseValues(indices=self.sparse_indices_1, values=self.sparse_values_1)),
                           Vector(id='vec2', values=self.vals2, metadata=dict_to_proto_struct(self.md2),
                                  sparse_values=SparseValues(indices=self.sparse_indices_2, values=self.sparse_values_2))],
                          namespace='ns')
        self._assert_called_once([self.expected_vec_md_sparse1, self.expected_vec_md_sparse2])

    def test_upsert_dict(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        dict1 = {'id': 'vec1', 'values': self.vals1}
        dict2 = {'id': 'vec2', 'values': self.vals2}
        self.index.upsert([dict1, dict2], namespace='ns')
        self._assert_called_once([
            self.expected_vec1,
            self.expected_vec2]
        )

    def test_upsert_dict_md(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        dict1 = {'id': 'vec1', 'values': self.vals1, 'metadata': self.md1}
        dict2 = {'id': 'vec2', 'values': self.vals2, 'metadata': self.md2}
        self.index.upsert([dict1, dict2], namespace='ns')
        self._assert_called_once([
            self.expected_vec_md1,
            self.expected_vec_md2]
        )

    def test_upsert_dict_sparse(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        dict1 = {'id': 'vec1', 'values': self.vals1,
                 'sparse_values': {'indices': self.sparse_indices_1, 'values': self.sparse_values_1}}
        dict2 = {'id': 'vec2', 'values': self.vals2,
                 'sparse_values': {'indices': self.sparse_indices_2, 'values': self.sparse_values_2}}
        self.index.upsert([dict1, dict2], namespace='ns')
        self._assert_called_once([
            Vector(id='vec1', values=self.vals1, metadata={},
                   sparse_values=SparseValues(indices=self.sparse_indices_1, values=self.sparse_values_1)),
            Vector(id='vec2', values=self.vals2, metadata={},
                   sparse_values=SparseValues(indices=self.sparse_indices_2, values=self.sparse_values_2))]
        )

    def test_upsert_dict_sparse_md(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        dict1 = {'id': 'vec1', 'values': self.vals1,
                 'sparse_values': {'indices': self.sparse_indices_1, 'values': self.sparse_values_1},
                 'metadata': self.md1}
        dict2 = {'id': 'vec2', 'values': self.vals2,
                 'sparse_values': {'indices': self.sparse_indices_2, 'values': self.sparse_values_2},
                 'metadata': self.md2}
        self.index.upsert([dict1, dict2], namespace='ns')
        self._assert_called_once([
            Vector(id='vec1', values=self.vals1, metadata=dict_to_proto_struct(self.md1),
                   sparse_values=SparseValues(indices=self.sparse_indices_1, values=self.sparse_values_1)),
            Vector(id='vec2', values=self.vals2, metadata=dict_to_proto_struct(self.md2),
                   sparse_values=SparseValues(indices=self.sparse_indices_2, values=self.sparse_values_2))]
        )

    def test_upsert_dict_negative(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)

        # Missing required keys
        dict1 = {'values': self.vals1}
        dict2 = {'id': 'vec2'}
        with pytest.raises(ValueError):
            self.index.upsert([dict1, dict2])
        with pytest.raises(ValueError):
            self.index.upsert([dict1])
        with pytest.raises(ValueError):
            self.index.upsert([dict2])

        # Excess keys
        dict1 = {'id': 'vec1', 'values': self.vals1}
        dict2 = {'id': 'vec2', 'values': self.vals2, 'animal': 'dog'}
        with pytest.raises(ValueError) as e:
            self.index.upsert([dict1, dict2])
            assert 'animal' in str(e.value)

        dict1 = {'id': 'vec1', 'values': self.vals1, 'metadatta': self.md2}
        dict2 = {'id': 'vec2', 'values': self.vals2}
        with pytest.raises(ValueError) as e:
            self.index.upsert([dict1, dict2])
            assert 'metadatta' in str(e.value)

    @pytest.mark.parametrize("key,new_val", [
        ("id", 4.2),
        ("id", ['vec1']),
        ("values", ['the', 'lazy', 'fox']),
        ("values", 'the lazy fox'),
        ("values", 0.5),
        ("metadata", np.nan),
        ("metadata", ['key1', 'key2']),
        ("sparse_values", 'cat'),
        ("sparse_values", []),
    ])
    def test_upsert_dict_negative_types(self, mocker, key, new_val):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)

        full_dict1 = {'id': 'vec1', 'values': self.vals1,
                      'sparse_values': {'indices': self.sparse_indices_1, 'values': self.sparse_values_1},
                      'metadata': self.md1}

        dict1 = deepcopy(full_dict1)
        dict1[key] = new_val
        with pytest.raises(TypeError) as e:
            self.index.upsert([dict1])
        assert key in str(e.value)

    @pytest.mark.parametrize("key,new_val", [
        ("indices", 3),
        ("indices", [1.2, 0.5]),
        ("values", ['1', '4.4']),
        ("values", 0.5),
    ])
    def test_upsert_dict_negative_types_sparse(self, mocker, key, new_val):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)

        full_dict1 = {'id': 'vec1', 'values': self.vals1,
                      'sparse_values': {'indices': self.sparse_indices_1, 'values': self.sparse_values_1},
                      'metadata': self.md1}

        dict1 = deepcopy(full_dict1)
        dict1['sparse_values'][key] = new_val
        with pytest.raises(TypeError) as e:
            self.index.upsert([dict1])
        assert 'sparse' in str(e.value)
        assert key in str(e.value)

    def test_upsert_dataframe(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True,
                            side_effect=lambda stub, upsert_request, timeout: MockUpsertDelegate(UpsertResponse(
                                upserted_count=len(upsert_request.vectors))))
        df = pd.DataFrame([
            {'id': 'vec1', 'values': self.vals1,
                 'sparse_values': {'indices': self.sparse_indices_1, 'values': self.sparse_values_1},
                 'metadata': self.md1},
            {'id': 'vec2', 'values': self.vals2,
                 'sparse_values': {'indices': self.sparse_indices_2, 'values': self.sparse_values_2},
                 'metadata': self.md2}
        ])
        self.index.upsert_from_dataframe(df, namespace='ns')
        self._assert_called_once([self.expected_vec_md_sparse1, self.expected_vec_md_sparse2],
            async_call=True
        )


    def test_upsert_dataframe_sync(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True,
                            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                                upserted_count=len(upsert_request.vectors)))
        df = pd.DataFrame([
            {'id': 'vec1', 'values': self.vals1,
                 'sparse_values': {'indices': self.sparse_indices_1, 'values': self.sparse_values_1},
                 'metadata': self.md1},
            {'id': 'vec2', 'values': self.vals2,
                 'sparse_values': {'indices': self.sparse_indices_2, 'values': self.sparse_values_2},
                 'metadata': self.md2}
        ])
        self.index.upsert_from_dataframe(df, namespace='ns', use_async_requests=False)
        self._assert_called_once([self.expected_vec_md_sparse1, self.expected_vec_md_sparse2],
            async_call=False
        )

    def test_upsert_dataframe_negative(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        full_dict1 = {'id': 'vec1', 'values': self.vals1,
                      'sparse_values': {'indices': self.sparse_indices_1, 'values': self.sparse_values_1},
                      'metadata': self.md1}
        full_df = pd.DataFrame([full_dict1])

        # Not a DF
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe([full_dict1])
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe(full_dict1)

        # Missing Cols
        df = full_df.copy()
        df.drop(columns=['id'], inplace=True)
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe(df)

        # Excess cols
        df = full_df.copy()
        df['animals'] = ['dog']
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe(df)

        df = full_df.copy()
        df['metadat'] = df['metadata']
        with pytest.raises(ValueError):
            self.index.upsert_from_dataframe(df)

    def test_upsert_async_upsertInputVectorsAsync(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.upsert([self.expected_vec_md1,
                           self.expected_vec_md2],
                          namespace='ns',
                          async_req=True)
        self._assert_called_once([
            self.expected_vec_md1,
            self.expected_vec_md2],
            async_call=True
        )

    def test_upsert_vectorListIsMultiplyOfBatchSize_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True,
                            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                                upserted_count=len(upsert_request.vectors)))

        result = self.index.upsert([self.expected_vec_md1,
                                    self.expected_vec_md2],
                                   namespace='ns',
                                   batch_size=1,
                                   show_progress=False)
        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(
                vectors=[
                    Vector(id='vec1', values=self.vals1, metadata=dict_to_proto_struct(self.md1))],
                namespace='ns'),
            timeout=None)

        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(
                vectors=[self.expected_vec_md2],
                namespace='ns'),
            timeout=None)

        assert result.upserted_count == 2

    def test_upsert_vectorListNotMultiplyOfBatchSize_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True,
                            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                                upserted_count=len(upsert_request.vectors)))

        result = self.index.upsert([self.expected_vec_md1,
                                    Vector(id='vec2', values=self.vals2, metadata=dict_to_proto_struct(self.md2)),
                                    Vector(id='vec3', values=self.vals1, metadata=dict_to_proto_struct(self.md1))],
                                   namespace='ns',
                                   batch_size=2)
        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(
                vectors=[
                    self.expected_vec_md1,
                    self.expected_vec_md2],
                namespace='ns'),
            timeout=None)

        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(
                vectors=[Vector(id='vec3', values=self.vals1, metadata=dict_to_proto_struct(self.md1))],
                namespace='ns'),
            timeout=None)

        assert result.upserted_count == 3

    def test_upsert_vectorListSmallerThanBatchSize_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True,
                            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                                upserted_count=len(upsert_request.vectors)))

        result = self.index.upsert([self.expected_vec_md1,
                                    self.expected_vec_md2],
                                   namespace='ns',
                                   batch_size=5)
        self._assert_called_once([
                    self.expected_vec_md1,
                    self.expected_vec_md2],
        )

        assert result.upserted_count == 2

    def test_upsert_tuplesList_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True,
                            side_effect=lambda stub, upsert_request, timeout: UpsertResponse(
                                upserted_count=len(upsert_request.vectors)))

        result = self.index.upsert([('vec1', self.vals1, self.md1),
                                    ('vec2', self.vals2, self.md2),
                                    ('vec3', self.vals1, self.md1)],
                                   namespace='ns',
                                   batch_size=2)
        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(
                vectors=[
                    self.expected_vec_md1,
                    self.expected_vec_md2],
                namespace='ns'),
            timeout=None)

        self.index._wrap_grpc_call.assert_any_call(
            self.index.stub.Upsert,
            UpsertRequest(
                vectors=[Vector(id='vec3', values=self.vals1, metadata=dict_to_proto_struct(self.md1))],
                namespace='ns'),
            timeout=None)

        assert result.upserted_count == 3

    def test_upsert_batchSizeIsNotPositive_errorIsRaised(self):
        with pytest.raises(ValueError, match='batch_size must be a positive integer'):
            self.index.upsert([Vector(id='vec1', values=self.vals1, metadata=dict_to_proto_struct(self.md1))],
                              namespace='ns',
                              batch_size=0)

        with pytest.raises(ValueError, match='batch_size must be a positive integer'):
            self.index.upsert([Vector(id='vec1', values=self.vals1, metadata=dict_to_proto_struct(self.md1))],
                              namespace='ns',
                              batch_size=-1)

    def test_upsert_useBatchSizeAndAsyncReq_valueErrorRaised(self):
        with pytest.raises(ValueError, match='async_req is not supported when batch_size is provided.'):
            self.index.upsert([Vector(id='vec1', values=self.vals1, metadata=dict_to_proto_struct(self.md1))],
                              namespace='ns',
                              batch_size=2,
                              async_req=True)

    # endregion

    # region: query tests

    def test_query_byVectorNoFilter_queryVectorNoFilter(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.query(top_k=10, vector=self.vals1)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, vector=self.vals1),
            timeout=None,
        )

    def test_query_byVectorWithFilter_queryVectorWithFilter(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.query(top_k=10, vector=self.vals1, filter=self.filter1, namespace='ns', timeout=10)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, vector=self.vals1, filter=dict_to_proto_struct(self.filter1), namespace='ns'),
            timeout=10,
        )

    def test_query_byTuplesNoFilter_queryVectorsNoFilter(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.query(top_k=10, queries=[
            (self.vals1,),
            (self.vals2,)
        ])
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(queries=[
                QueryVector(values=self.vals1, filter={}),
                QueryVector(values=self.vals2, filter={})
            ], top_k=10),
            timeout=None,
        )

    def test_query_byTuplesWithFilter_queryVectorsWithFilter(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.query(top_k=10, queries=[
            (self.vals1, self.filter1),
            (self.vals2, self.filter2)
        ])
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(queries=[
                QueryVector(values=self.vals1, filter=dict_to_proto_struct(self.filter1)),
                QueryVector(values=self.vals2, filter=dict_to_proto_struct(self.filter2))
            ], top_k=10),
            timeout=None,
        )

    def test_query_byVecId_queryByVecId(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.query(top_k=10, id='vec1', include_metadata=True, include_values=False)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Query,
            QueryRequest(top_k=10, id='vec1', include_metadata=True, include_values=False),
            timeout=None,
        )

    # endregion

    # region: delete tests

    def test_delete_byIds_deleteByIds(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.delete(ids=['vec1', 'vec2'])
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Delete,
            DeleteRequest(ids=['vec1', 'vec2']),
            timeout=None,
        )

    def test_delete_byIdsAsync_deleteByIdsAsync(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.delete(ids=['vec1', 'vec2'], async_req=True)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Delete.future,
            DeleteRequest(ids=['vec1', 'vec2']),
            timeout=None,
        )

    def test_delete_deleteAllByFilter_deleteAllByFilter(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.delete(delete_all=True, filter=self.filter1, namespace='ns', timeout=30)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Delete,
            DeleteRequest(delete_all=True, filter=dict_to_proto_struct(self.filter1), namespace='ns'),
            timeout=30,
        )

    def test_delete_deleteAllNoFilter_deleteNoFilter(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.delete(delete_all=True)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Delete,
            DeleteRequest(delete_all=True),
            timeout=None,
        )

    # endregion

    # region: fetch tests

    def test_fetch_byIds_fetchByIds(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.fetch(['vec1', 'vec2'])
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Fetch,
            FetchRequest(ids=['vec1', 'vec2']),
            timeout=None,
        )

    def test_fetch_byIdsAndNS_fetchByIdsAndNS(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.fetch(['vec1', 'vec2'], namespace='ns', timeout=30)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Fetch,
            FetchRequest(ids=['vec1', 'vec2'], namespace='ns'),
            timeout=30,
        )

    # endregion

    # region: update tests

    def test_update_byIdAnValues_updateByIdAndValues(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.update(id='vec1', values=self.vals1, namespace='ns', timeout=30)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(id='vec1', values=self.vals1, namespace='ns'),
            timeout=30,
        )

    def test_update_byIdAnValuesAsync_updateByIdAndValuesAsync(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.update(id='vec1', values=self.vals1, namespace='ns', timeout=30, async_req=True)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Update.future,
            UpdateRequest(id='vec1', values=self.vals1, namespace='ns'),
            timeout=30,
        )

    def test_update_byIdAnValuesAndMetadata_updateByIdAndValuesAndMetadata(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.update('vec1', values=self.vals1, set_metadata=self.md1)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.Update,
            UpdateRequest(id='vec1', values=self.vals1, set_metadata=dict_to_proto_struct(self.md1)),
            timeout=None,
        )

    # endregion

    # region: describe index tests

    def test_describeIndexStats_callWithoutFilter_CalledWithoutFilter(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.describe_index_stats()
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.DescribeIndexStats,
            DescribeIndexStatsRequest(),
            timeout=None,
        )

    def test_describeIndexStats_callWithFilter_CalledWithFilter(self, mocker):
        mocker.patch.object(self.index, '_wrap_grpc_call', autospec=True)
        self.index.describe_index_stats(filter=self.filter1)
        self.index._wrap_grpc_call.assert_called_once_with(
            self.index.stub.DescribeIndexStats,
            DescribeIndexStatsRequest(filter=dict_to_proto_struct(self.filter1)),
            timeout=None,
        )

    # endregion

class MockUpsertDelegate:
    def __init__(self, upsert_response: UpsertResponse):
        self.response = upsert_response

    def result(self, timeout):
        return self.response