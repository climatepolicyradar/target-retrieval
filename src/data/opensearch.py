from typing import Optional

from opensearchpy import OpenSearch


def _convert_to_bool(x):
    if x.lower() == "true":
        return True
    elif x.lower() == "false":
        return False


class OpenSearchIndex:
    """Methods to connect to OpenSearch instance, define an index mapping, and load data into an index."""

    def __init__(
        self,
        index_name: str,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        opensearch_connector_kwargs: dict = {},
    ):
        self.index_name = index_name

        self._url = url
        self._login = (username, password)
        self._opensearch_connector_kwargs = opensearch_connector_kwargs

        self._connect_to_opensearch()

    def _connect_to_opensearch(
        self,
    ):

        self.opns = OpenSearch(
            [self._url],
            http_auth=self._login,
            **self._opensearch_connector_kwargs,
        )

    def is_connected(self) -> bool:
        """Check if we are connected to the OpenSearch instance."""
        return self.opns.ping()
