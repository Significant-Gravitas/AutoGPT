import json

import pytest
from urllib3_mock import Responses
import responses as req_responses

import pinecone
from pinecone import ApiTypeError, ApiException

responses_req = Responses()
responses = Responses('requests.packages.urllib3')

@req_responses.activate
@responses.activate
def test_invalid_upsert_request_vector_value_type():
    environment = 'example-environment'
    project_name = 'example-project'
    req_responses.add(
        'GET', f'https://controller.{environment}.pinecone.io/actions/whoami',
        status=200, content_type='application/json',
        body=json.dumps(dict(project_name=project_name, user_label='example-label', user_name='test'))
    )
    responses.add(
        'POST', '/vectors/upsert',
        status=400, content_type='text/plain',
        adding_headers={
            'content-length': '62',
            'date': 'Thu, 28 Oct 2021 09:14:51 GMT',
            'server': 'envoy',
            'connection': 'close'
        },
        body='vectors[0].values[1]: invalid value "type" for type TYPE_FLOAT'
    )

    pinecone.init('example-api-key', environment='example-environment')
    with pytest.raises(ApiException) as exc_info:
        index = pinecone.Index('example-index')
        resp = index.upsert(vectors=[('vec1', [0.1]*8), ('vec2', [0.2]*8)])

    assert len(responses.calls) == 1
    assert responses.calls[0].request.scheme == 'https'
    assert responses.calls[0].request.host == 'example-index-example-project.svc.example-environment.pinecone.io'
    assert responses.calls[0].request.url == '/vectors/upsert'


@req_responses.activate
@responses.activate
def test_multiple_indexes():
    environment = 'example-environment'
    project_name = 'example-project'
    index1_name = 'index-1'
    index2_name = 'index-2'
    req_responses.add(
        'GET', f'https://controller.{environment}.pinecone.io/actions/whoami',
        status=200, content_type='application/json',
        body=json.dumps(dict(project_name=project_name, user_label='example-label', user_name='test'))
    )
    responses.add(
        'GET', f'/describe_index_stats',
        status=200, content_type='application/json',
        adding_headers={
            'date': 'Thu, 28 Oct 2021 09:14:51 GMT',
            'server': 'envoy'
        },
        body='{"namespaces":{"":{"vectorCount":50000},"example-namespace-2":{"vectorCount":30000}},"dimension":1024}'
    )

    pinecone.init('example-api-key', environment='example-environment')

    index1 = pinecone.Index(index1_name)
    resp1 = index1.describe_index_stats()
    assert resp1.dimension == 1024
    assert responses.calls[0].request.host == f'{index1_name}-{project_name}.svc.{environment}.pinecone.io'

    index2 = pinecone.Index(index2_name)
    resp2 = index2.describe_index_stats()
    assert resp2.dimension == 1024
    assert responses.calls[1].request.host == f'{index2_name}-{project_name}.svc.{environment}.pinecone.io'


@req_responses.activate
@responses.activate
def test_invalid_delete_response_unrecognized_field():
    # unrecognized response fields are okay, shouldn't raise an exception
    environment = 'example-environment'
    project_name = 'example-project'
    req_responses.add(
        'GET', f'https://controller.{environment}.pinecone.io/actions/whoami',
        status=200, content_type='application/json',
        body=json.dumps(dict(project_name=project_name, user_label='example-label', user_name='test'))
    )
    responses.add(
        'DELETE', '/vectors/delete',
        body='{"unexpected_key": "xyzzy"}',
        status=200, content_type='application/json'
    )

    pinecone.init('example-api-key', environment=environment)
    index = pinecone.Index('example-index')
    resp = index.delete(ids=['vec1', 'vec2'])

    assert len(req_responses.calls) == 1
    assert responses.calls[0].request.scheme == 'https'
    assert responses.calls[0].request.host == f'example-index-{project_name}.svc.{environment}.pinecone.io'
    assert responses.calls[0].request.url == '/vectors/delete?ids=vec1&ids=vec2'


@responses.activate
def test_delete_response_missing_field():
    # missing (optional) response fields are okay, shouldn't raise an exception
    pinecone.init('example-api-key', environment='example-environment')
    responses.add('DELETE', '/vectors/delete',
                  body='{}',
                  status=200, content_type='application/json')
    index = pinecone.Index('example-index')
    # this should not raise
    index.delete(ids=['vec1', 'vec2'])



@responses.activate
def _test_invalid_delete_response_wrong_type():
    # FIXME: re-enable this test when accepted_count added back to response
    # wrong-typed response fields should raise an exception
    pinecone.init('example-api-key', environment='example-environment')

    responses.add('DELETE', '/vectors/delete',
                  body='{"deleted_count": "foobar"}',
                  status=200, content_type='application/json')

    index = pinecone.Index('example-index')

    with pytest.raises(ApiTypeError) as exc_info:
        resp = index.delete(ids=['vec1', 'vec2'])
        assert resp.deleted_count == 2
