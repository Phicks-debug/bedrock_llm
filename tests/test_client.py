"""Unit tests for the Client class."""
import unittest
from unittest.mock import ANY, MagicMock, patch

import boto3
from botocore.config import Config

from bedrock_llm.aws_clients import AWSClientManager
from bedrock_llm.client import Client
from bedrock_llm.config.base import RetryConfig
from bedrock_llm.types.enums import ModelName
from bedrock_llm.models.anthropic import ClaudeImplementation
from bedrock_llm.models.meta import LlamaImplementation
from bedrock_llm.models.amazon import TitanImplementation


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.region_name = "us-west-2"
        self.model_name = ModelName.CLAUDE_3_HAIKU
        
        # Clear the cached clients and implementations
        Client._model_implementations = {}
        AWSClientManager._sync_clients = {}
        AWSClientManager._async_clients = {}

    @patch('boto3.Session')
    def test_client_initialization(self, mock_session):
        """Test basic client initialization."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        client = Client(self.region_name, self.model_name)
        
        self.assertEqual(client.region_name, self.region_name)
        self.assertEqual(client.model_name, self.model_name)
        self.assertIsInstance(client.retry_config, RetryConfig)
        self.assertIsInstance(client.model_implementation, ClaudeImplementation)

    def test_model_implementation_caching(self):
        """Test that model implementations are properly cached."""
        client1 = Client(self.region_name, ModelName.CLAUDE_3_HAIKU)
        client2 = Client(self.region_name, ModelName.CLAUDE_3_HAIKU)
        
        # Both clients should have the same implementation instance
        self.assertIs(client1.model_implementation, client2.model_implementation)

    @patch('boto3.Session')
    def test_bedrock_client_caching(self, mock_session):
        """Test that Bedrock clients are properly cached."""
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client

        client1 = Client(self.region_name, self.model_name)
        client2 = Client(self.region_name, self.model_name)
        
        # Both clients should have the same Bedrock client instance
        self.assertIs(client1._sync_client, client2._sync_client)
        
        # boto3.client should only be called once
        mock_session.return_value.client.assert_called_once()

    def test_different_model_implementations(self):
        """Test that different models get different implementations."""
        claude_client = Client(self.region_name, ModelName.CLAUDE_3_HAIKU)
        llama_client = Client(self.region_name, ModelName.LLAMA_3_2_1B)
        titan_client = Client(self.region_name, ModelName.TITAN_LITE)
        nova_client = Client(self.region_name, ModelName.NOVA_MICRO)
        
        self.assertIsInstance(claude_client.model_implementation, ClaudeImplementation)
        self.assertIsInstance(llama_client.model_implementation, LlamaImplementation)
        self.assertIsInstance(titan_client.model_implementation, TitanImplementation)
        self.assertIsInstance(nova_client.model_implementation, TitanImplementation)

    @patch('bedrock_llm.aws_clients.boto3.Session')
    def test_custom_profile(self, mock_session):
        """Test client initialization with custom AWS profile."""
        profile_name = "custom-profile"
        mock_client = MagicMock()
        mock_session.return_value.client.return_value = mock_client
        
        client = Client(self.region_name, self.model_name, profile_name=profile_name)
        
        mock_session.assert_called_once_with(profile_name=profile_name)
        mock_session.return_value.client.assert_called_once_with(
            "bedrock-runtime",
            region_name=self.region_name,
            config=ANY,
        )

    def test_retry_config(self):
        """Test custom retry configuration."""
        custom_retry = RetryConfig(max_retries=5, retry_delay=2.0)
        client = Client(self.region_name, self.model_name, retry_config=custom_retry)
        
        self.assertEqual(client.retry_config.max_retries, 5)
        self.assertEqual(client.retry_config.retry_delay, 2.0)


if __name__ == '__main__':
    unittest.main()
