import pandas as pd
import pytest

from pinecone.core.client.api_client import Endpoint

import pinecone
from pinecone import DescribeIndexStatsRequest, ScoredVector, QueryResponse, UpsertResponse, SparseValues


class TestRestIndex:

    def setup_method(self):
        self.vector_dim = 8
        self.id1 = 'vec1'
        self.id2 = 'vec2'
        self.vals1 = [0.1] * self.vector_dim
        self.vals2 = [0.2] * self.vector_dim
        self.md1 = {'genre': 'action', 'year': 2021}
        self.md2 = {'genre': 'documentary', 'year': 2020}
        self.filter1 = {'genre': {'$in': ['action']}}
        self.filter2 = {'year': {'$eq': 2020}}
        self.svi1 = [1, 3, 5]
        self.svv1 = [0.1, 0.2, 0.3]
        self.sv1 = {
            'indices': self.svi1,
            'values': self.svv1,
        }
        self.svi2 = [2, 4, 6]
        self.svv2 = [0.1, 0.2, 0.3]
        self.sv2 = {
            'indices': self.svi2,
            'values': self.svv2
        }

        pinecone.init(api_key='example-key')
        self.index = pinecone.Index('example-name')

    # region: upsert tests

    def test_upsert_tuplesOfIdVec_UpserWithoutMD(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True)
        self.index.upsert([('vec1', self.vals1), ('vec2', self.vals2)], namespace='ns')
        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata={}),
                pinecone.Vector(id='vec2', values=self.vals2, metadata={})
            ], namespace='ns')
        )

    def test_upsert_tuplesOfIdVecMD_UpsertVectorsWithMD(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True)
        self.index.upsert([('vec1', self.vals1, self.md1),
                           ('vec2', self.vals2, self.md2)])
        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)
            ])
        )

    def test_upsert_dictOfIdVecMD_UpsertVectorsWithMD(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True)
        self.index.upsert([{'id': self.id1, 'values': self.vals1, 'metadata': self.md1},
                           {'id': self.id2, 'values': self.vals2, 'metadata': self.md2}])
        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)
            ])
        )

    def test_upsert_dictOfIdVecMD_UpsertVectorsWithoutMD(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True)
        self.index.upsert([{'id': self.id1, 'values': self.vals1},
                           {'id': self.id2, 'values': self.vals2}])
        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1),
                pinecone.Vector(id='vec2', values=self.vals2)
            ])
        )

    def test_upsert_dictOfIdVecMD_UpsertVectorsWithSparseValues(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True)
        self.index.upsert([{'id': self.id1, 'values': self.vals1, 'sparse_values': self.sv1},
                           {'id': self.id2, 'values': self.vals2, 'sparse_values': self.sv2}])
        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, sparse_values=SparseValues(**self.sv1)),
                pinecone.Vector(id='vec2', values=self.vals2, sparse_values=SparseValues(**self.sv2))
            ])
        )

    def test_upsert_vectors_upsertInputVectors(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True)
        self.index.upsert(vectors=[
            pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
            pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)],
            namespace='ns')
        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)
            ], namespace='ns')
        )

    def test_upsert_parallelUpsert_callUpsertParallel(self, mocker):
        mocker.patch.object(Endpoint, '__call__', autospec=True)
        chunks = [[pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1)],
                  [pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)]]
        with pinecone.Index('example-index', pool_threads=30) as index:
            # Send requests in parallel
            async_results = [
                index.upsert(vectors=ids_vectors_chunk, namespace="ns", async_req=True)
                for ids_vectors_chunk in chunks
            ]
            # Wait for and retrieve responses (this raises in case of error)
            [async_result.get() for async_result in async_results]

            Endpoint.__call__.assert_any_call(
                index._vector_api.upsert,
                pinecone.UpsertRequest(vectors=[
                    pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                ],
                    namespace='ns'),
                async_req=True
            )

            Endpoint.__call__.assert_any_call(
                index._vector_api.upsert,
                pinecone.UpsertRequest(vectors=[
                    pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2),
                ],
                    namespace='ns'),
                async_req=True
            )

    def test_upsert_vectorListIsMultiplyOfBatchSize_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(self.index._vector_api,
                            'upsert',
                            autospec=True,
                            side_effect=lambda upsert_request: UpsertResponse(
                                upserted_count=len(upsert_request.vectors)))

        result = self.index.upsert(vectors=[
            pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
            pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)],
            namespace='ns',
            batch_size=1,
            show_progress=False)

        self.index._vector_api.upsert.assert_any_call(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
            ], namespace='ns')
        )

        self.index._vector_api.upsert.assert_any_call(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2),
            ], namespace='ns')
        )

        assert result.upserted_count == 2

    def test_upsert_vectorListNotMultiplyOfBatchSize_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(self.index._vector_api,
                            'upsert',
                            autospec=True,
                            side_effect=lambda upsert_request: UpsertResponse(
                                upserted_count=len(upsert_request.vectors)))

        result = self.index.upsert(vectors=[
            pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
            pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2),
            pinecone.Vector(id='vec3', values=self.vals1, metadata=self.md1)],
            namespace='ns',
            batch_size=2)

        self.index._vector_api.upsert.assert_any_call(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2),
            ], namespace='ns')
        )

        self.index._vector_api.upsert.assert_any_call(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec3', values=self.vals1, metadata=self.md1),
            ], namespace='ns')
        )

        assert result.upserted_count == 3

    def test_upsert_vectorListSmallerThanBatchSize_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(self.index._vector_api,
                            'upsert',
                            autospec=True,
                            side_effect=lambda upsert_request: UpsertResponse(
                                upserted_count=len(upsert_request.vectors)))

        result = self.index.upsert(vectors=[
            pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
            pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2),
            pinecone.Vector(id='vec3', values=self.vals1, metadata=self.md1)],
            namespace='ns',
            batch_size=5)

        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2),
                pinecone.Vector(id='vec3', values=self.vals1, metadata=self.md1),
            ], namespace='ns')
        )

        assert result.upserted_count == 3

    def test_upsert_tuplesList_vectorsUpsertedInBatches(self, mocker):
        mocker.patch.object(self.index._vector_api,
                            'upsert',
                            autospec=True,
                            side_effect=lambda upsert_request: UpsertResponse(
                                upserted_count=len(upsert_request.vectors)))

        result = self.index.upsert(vectors=
                                   [('vec1', self.vals1, self.md1),
                                    ('vec2', self.vals2, self.md2),
                                    ('vec3', self.vals1, self.md1)],
                                   namespace='ns',
                                   batch_size=2)

        self.index._vector_api.upsert.assert_any_call(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2),
            ], namespace='ns')
        )

        self.index._vector_api.upsert.assert_any_call(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec3', values=self.vals1, metadata=self.md1),
            ], namespace='ns')
        )

        assert result.upserted_count == 3

    def test_upsert_dataframe(self, mocker):
        mocker.patch.object(self.index._vector_api, 'upsert', autospec=True, return_value=UpsertResponse(upserted_count=2))
        df = pd.DataFrame([
            {'id': self.id1, 'values': self.vals1, 'metadata': self.md1},
            {'id': self.id2, 'values': self.vals2, 'metadata': self.md2}
        ])
        self.index.upsert_from_dataframe(df)

        self.index._vector_api.upsert.assert_called_once_with(
            pinecone.UpsertRequest(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1),
                pinecone.Vector(id='vec2', values=self.vals2, metadata=self.md2)
            ])
        )

    def test_upsert_batchSizeIsNotPositive_errorIsRaised(self):
        with pytest.raises(ValueError, match='batch_size must be a positive integer'):
            self.index.upsert(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1)],
                namespace='ns',
                batch_size=0)

        with pytest.raises(ValueError, match='batch_size must be a positive integer'):
            self.index.upsert(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1)],
                namespace='ns',
                batch_size=-1)

    def test_upsert_useBatchSizeAndAsyncReq_valueErrorRaised(self):
        with pytest.raises(ValueError, match='async_req is not supported when batch_size is provided.'):
            self.index.upsert(vectors=[
                pinecone.Vector(id='vec1', values=self.vals1, metadata=self.md1)],
                namespace='ns',
                batch_size=1,
                async_req=True)

    # endregion

    # region: query tests

    def test_query_byVectorNoFilter_queryVectorNoFilter(self, mocker):
        response = QueryResponse(results=[],
                                 matches=[ScoredVector(id="1",
                                                       score=0.9,
                                                       values=[0.0],
                                                       metadata={"a": 2})],
                                 namespace="test")

        mocker.patch.object(self.index._vector_api, 'query', autospec=True, return_value=response)

        actual = self.index.query(top_k=10, vector=self.vals1)

        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, vector=self.vals1)
        )
        expected = QueryResponse(matches=[ScoredVector(id="1",
                                                       score=0.9,
                                                       values=[0.0],
                                                       metadata={"a": 2})],
                                 namespace="test")
        expected._data_store.pop('results', None)
        assert actual == expected

    def test_query_byVectorWithFilter_queryVectorWithFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'query', autospec=True)
        self.index.query(top_k=10, vector=self.vals1, filter=self.filter1, namespace='ns')
        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, vector=self.vals1, filter=self.filter1, namespace='ns')
        )

    def test_query_byTuplesNoFilter_queryVectorsNoFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'query', autospec=True)
        self.index.query(top_k=10, queries=[
            (self.vals1,),
            (self.vals2,)
        ])
        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, queries=[
                pinecone.QueryVector(values=self.vals1),
                pinecone.QueryVector(values=self.vals2)
            ])
        )

    def test_query_byTuplesWithFilter_queryVectorsWithFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'query', autospec=True)
        self.index.query(top_k=10, queries=[
            (self.vals1, self.filter1),
            (self.vals2, self.filter2)
        ])
        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, queries=[
                pinecone.QueryVector(values=self.vals1, filter=self.filter1),
                pinecone.QueryVector(values=self.vals2, filter=self.filter2)
            ])
        )

    def test_query_byVecId_queryByVecId(self, mocker):
        mocker.patch.object(self.index._vector_api, 'query', autospec=True)
        self.index.query(top_k=10, id='vec1', include_metadata=True, include_values=False)
        self.index._vector_api.query.assert_called_once_with(
            pinecone.QueryRequest(top_k=10, id='vec1', include_metadata=True, include_values=False)
        )

    # endregion

    # region: delete tests

    def test_delete_byIds_deleteByIds(self, mocker):
        mocker.patch.object(self.index._vector_api, 'delete', autospec=True)
        self.index.delete(ids=['vec1', 'vec2'])
        self.index._vector_api.delete.assert_called_once_with(
            pinecone.DeleteRequest(ids=['vec1', 'vec2'])
        )

    def test_delete_deleteAllByFilter_deleteAllByFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'delete', autospec=True)
        self.index.delete(delete_all=True, filter=self.filter1, namespace='ns')
        self.index._vector_api.delete.assert_called_once_with(
            pinecone.DeleteRequest(delete_all=True, filter=self.filter1, namespace='ns')
        )

    def test_delete_deleteAllNoFilter_deleteNoFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'delete', autospec=True)
        self.index.delete(delete_all=True)
        self.index._vector_api.delete.assert_called_once_with(
            pinecone.DeleteRequest(delete_all=True)
        )

    # endregion

    # region: fetch tests

    def test_fetch_byIds_fetchByIds(self, mocker):
        mocker.patch.object(self.index._vector_api, 'fetch', autospec=True)
        self.index.fetch(ids=['vec1', 'vec2'])
        self.index._vector_api.fetch.assert_called_once_with(
            ids=['vec1', 'vec2']
        )

    def test_fetch_byIdsAndNS_fetchByIdsAndNS(self, mocker):
        mocker.patch.object(self.index._vector_api, 'fetch', autospec=True)
        self.index.fetch(ids=['vec1', 'vec2'], namespace='ns')
        self.index._vector_api.fetch.assert_called_once_with(
            ids=['vec1', 'vec2'], namespace='ns'
        )

    # endregion

    # region: update tests

    def test_update_byIdAnValues_updateByIdAndValues(self, mocker):
        mocker.patch.object(self.index._vector_api, 'update', autospec=True)
        self.index.update(id='vec1', values=self.vals1, namespace='ns')
        self.index._vector_api.update.assert_called_once_with(
            pinecone.UpdateRequest(id='vec1', values=self.vals1, namespace='ns')
        )

    def test_update_byIdAnValuesAndMetadata_updateByIdAndValuesAndMetadata(self, mocker):
        mocker.patch.object(self.index._vector_api, 'update', autospec=True)
        self.index.update('vec1', values=self.vals1, metadata=self.md1)
        self.index._vector_api.update.assert_called_once_with(
            pinecone.UpdateRequest(id='vec1', values=self.vals1, metadata=self.md1)
        )

    # endregion

    # region: describe index tests

    def test_describeIndexStats_callWithoutFilter_CalledWithoutFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'describe_index_stats', autospec=True)
        self.index.describe_index_stats()
        self.index._vector_api.describe_index_stats.assert_called_once_with(
            DescribeIndexStatsRequest())

    def test_describeIndexStats_callWithFilter_CalledWithFilter(self, mocker):
        mocker.patch.object(self.index._vector_api, 'describe_index_stats', autospec=True)
        self.index.describe_index_stats(filter=self.filter1)
        self.index._vector_api.describe_index_stats.assert_called_once_with(
            DescribeIndexStatsRequest(filter=self.filter1))

    # endregion
