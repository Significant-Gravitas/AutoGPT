# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AsyncIO iterators for paging through paged API methods.

These iterators simplify the process of paging through API responses
where the request takes a page token and the response is a list of results with
a token for the next page. See `list pagination`_ in the Google API Style Guide
for more details.

.. _list pagination:
    https://cloud.google.com/apis/design/design_patterns#list_pagination

API clients that have methods that follow the list pagination pattern can
return an :class:`.AsyncIterator`:

    >>> results_iterator = await client.list_resources()

Or you can walk your way through items and call off the search early if
you find what you're looking for (resulting in possibly fewer requests)::

    >>> async for resource in results_iterator:
    ...     print(resource.name)
    ...     if not resource.is_valid:
    ...         break

At any point, you may check the number of items consumed by referencing the
``num_results`` property of the iterator::

    >>> async for my_item in results_iterator:
    ...     if results_iterator.num_results >= 10:
    ...         break

When iterating, not every new item will send a request to the server.
To iterate based on each page of items (where a page corresponds to
a request)::

    >>> async for page in results_iterator.pages:
    ...     print('=' * 20)
    ...     print('    Page number: {:d}'.format(iterator.page_number))
    ...     print('  Items in page: {:d}'.format(page.num_items))
    ...     print('     First item: {!r}'.format(next(page)))
    ...     print('Items remaining: {:d}'.format(page.remaining))
    ...     print('Next page token: {}'.format(iterator.next_page_token))
    ====================
        Page number: 1
      Items in page: 1
         First item: <MyItemClass at 0x7f1d3cccf690>
    Items remaining: 0
    Next page token: eav1OzQB0OM8rLdGXOEsyQWSG
    ====================
        Page number: 2
      Items in page: 19
         First item: <MyItemClass at 0x7f1d3cccffd0>
    Items remaining: 18
    Next page token: None
"""

import abc

from google.api_core.page_iterator import Page


def _item_to_value_identity(iterator, item):
    """An item to value transformer that returns the item un-changed."""
    # pylint: disable=unused-argument
    # We are conforming to the interface defined by Iterator.
    return item


class AsyncIterator(abc.ABC):
    """A generic class for iterating through API list responses.

    Args:
        client(google.cloud.client.Client): The API client.
        item_to_value (Callable[google.api_core.page_iterator_async.AsyncIterator, Any]):
            Callable to convert an item from the type in the raw API response
            into the native object. Will be called with the iterator and a
            single item.
        page_token (str): A token identifying a page in a result set to start
            fetching results from.
        max_results (int): The maximum number of results to fetch.
    """

    def __init__(
        self,
        client,
        item_to_value=_item_to_value_identity,
        page_token=None,
        max_results=None,
    ):
        self._started = False
        self.__active_aiterator = None

        self.client = client
        """Optional[Any]: The client that created this iterator."""
        self.item_to_value = item_to_value
        """Callable[Iterator, Any]: Callable to convert an item from the type
            in the raw API response into the native object. Will be called with
            the iterator and a
            single item.
        """
        self.max_results = max_results
        """int: The maximum number of results to fetch."""

        # The attributes below will change over the life of the iterator.
        self.page_number = 0
        """int: The current page of results."""
        self.next_page_token = page_token
        """str: The token for the next page of results. If this is set before
            the iterator starts, it effectively offsets the iterator to a
            specific starting point."""
        self.num_results = 0
        """int: The total number of results fetched so far."""

    @property
    def pages(self):
        """Iterator of pages in the response.

        returns:
            types.GeneratorType[google.api_core.page_iterator.Page]: A
                generator of page instances.

        raises:
            ValueError: If the iterator has already been started.
        """
        if self._started:
            raise ValueError("Iterator has already started", self)
        self._started = True
        return self._page_aiter(increment=True)

    async def _items_aiter(self):
        """Iterator for each item returned."""
        async for page in self._page_aiter(increment=False):
            for item in page:
                self.num_results += 1
                yield item

    def __aiter__(self):
        """Iterator for each item returned.

        Returns:
            types.GeneratorType[Any]: A generator of items from the API.

        Raises:
            ValueError: If the iterator has already been started.
        """
        if self._started:
            raise ValueError("Iterator has already started", self)
        self._started = True
        return self._items_aiter()

    async def __anext__(self):
        if self.__active_aiterator is None:
            self.__active_aiterator = self.__aiter__()
        return await self.__active_aiterator.__anext__()

    async def _page_aiter(self, increment):
        """Generator of pages of API responses.

        Args:
            increment (bool): Flag indicating if the total number of results
                should be incremented on each page. This is useful since a page
                iterator will want to increment by results per page while an
                items iterator will want to increment per item.

        Yields:
            Page: each page of items from the API.
        """
        page = await self._next_page()
        while page is not None:
            self.page_number += 1
            if increment:
                self.num_results += page.num_items
            yield page
            page = await self._next_page()

    @abc.abstractmethod
    async def _next_page(self):
        """Get the next page in the iterator.

        This does nothing and is intended to be over-ridden by subclasses
        to return the next :class:`Page`.

        Raises:
            NotImplementedError: Always, this method is abstract.
        """
        raise NotImplementedError


class AsyncGRPCIterator(AsyncIterator):
    """A generic class for iterating through gRPC list responses.

    .. note:: The class does not take a ``page_token`` argument because it can
        just be specified in the ``request``.

    Args:
        client (google.cloud.client.Client): The API client. This unused by
            this class, but kept to satisfy the :class:`Iterator` interface.
        method (Callable[protobuf.Message]): A bound gRPC method that should
            take a single message for the request.
        request (protobuf.Message): The request message.
        items_field (str): The field in the response message that has the
            items for the page.
        item_to_value (Callable[GRPCIterator, Any]): Callable to convert an
            item from the type in the JSON response into a native object. Will
            be called with the iterator and a single item.
        request_token_field (str): The field in the request message used to
            specify the page token.
        response_token_field (str): The field in the response message that has
            the token for the next page.
        max_results (int): The maximum number of results to fetch.

    .. autoattribute:: pages
    """

    _DEFAULT_REQUEST_TOKEN_FIELD = "page_token"
    _DEFAULT_RESPONSE_TOKEN_FIELD = "next_page_token"

    def __init__(
        self,
        client,
        method,
        request,
        items_field,
        item_to_value=_item_to_value_identity,
        request_token_field=_DEFAULT_REQUEST_TOKEN_FIELD,
        response_token_field=_DEFAULT_RESPONSE_TOKEN_FIELD,
        max_results=None,
    ):
        super().__init__(client, item_to_value, max_results=max_results)
        self._method = method
        self._request = request
        self._items_field = items_field
        self._request_token_field = request_token_field
        self._response_token_field = response_token_field

    async def _next_page(self):
        """Get the next page in the iterator.

        Returns:
            Page: The next page in the iterator or :data:`None` if
                there are no pages left.
        """
        if not self._has_next_page():
            return None

        if self.next_page_token is not None:
            setattr(self._request, self._request_token_field, self.next_page_token)

        response = await self._method(self._request)

        self.next_page_token = getattr(response, self._response_token_field)
        items = getattr(response, self._items_field)
        page = Page(self, items, self.item_to_value, raw_page=response)

        return page

    def _has_next_page(self):
        """Determines whether or not there are more pages with results.

        Returns:
            bool: Whether the iterator has more pages.
        """
        if self.page_number == 0:
            return True

        # Note: intentionally a falsy check instead of a None check. The RPC
        # can return an empty string indicating no more pages.
        if self.max_results is not None:
            if self.num_results >= self.max_results:
                return False

        return True if self.next_page_token else False
