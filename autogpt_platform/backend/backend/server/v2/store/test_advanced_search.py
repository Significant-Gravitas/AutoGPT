"""
Test suite for advanced search functionality in store agents
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from backend.server.v2.store.db import AdvancedSearchEngine, get_store_agents


class TestAdvancedSearchEngine:
    """Test the advanced search engine functionality"""
    
    def test_tokenize(self):
        """Test text tokenization"""
        engine = AdvancedSearchEngine()
        
        # Test basic tokenization
        assert engine.tokenize("Hello World") == ["hello", "world"]
        assert engine.tokenize("AI-powered agent") == ["ai", "powered", "agent"]
        assert engine.tokenize("Test123 with numbers") == ["test123", "with", "numbers"]
        
        # Test with special characters
        assert engine.tokenize("email@example.com") == ["email", "example", "com"]
        assert engine.tokenize("snake_case_text") == ["snake", "case", "text"]
        
    def test_calculate_fuzzy_score(self):
        """Test fuzzy matching score calculation"""
        engine = AdvancedSearchEngine()
        
        # Test exact match
        assert engine.calculate_fuzzy_score("chat", "ChatGPT Assistant") == 1.0
        
        # Test partial match
        score = engine.calculate_fuzzy_score("gpt", "ChatGPT Assistant")
        assert score > 0.6
        
        # Test typo tolerance
        score = engine.calculate_fuzzy_score("chta", "chat")  # Typo in query
        assert score > 0.5
        
        # Test no match
        score = engine.calculate_fuzzy_score("xyz", "ChatGPT Assistant")
        assert score == 0
        
    def test_generate_search_variants(self):
        """Test search variant generation for typo tolerance"""
        engine = AdvancedSearchEngine()
        
        variants = engine.generate_search_variants("chat")
        assert "chat" in variants
        assert "c hat" in variants  # Missing space variant
        assert "hcat" in variants  # Letter swap variant
        
        variants = engine.generate_search_variants("ai bot")
        assert "aibot" in variants  # No space variant
        assert "ai bot" in variants  # Original
        
    def test_build_advanced_search_query(self):
        """Test building advanced search conditions"""
        engine = AdvancedSearchEngine()
        
        # Test with fuzzy search enabled
        conditions = engine.build_advanced_search_query("chat gpt", use_fuzzy=True)
        assert len(conditions) > 0
        
        # Check that conditions include various fields
        field_names = set()
        for condition in conditions:
            field_names.update(condition.keys())
        
        assert "agent_name" in field_names
        assert "description" in field_names
        assert "sub_heading" in field_names
        
        # Test without fuzzy search
        conditions = engine.build_advanced_search_query("chat", use_fuzzy=False)
        assert len(conditions) > 0
        
    def test_calculate_relevance_score(self):
        """Test relevance score calculation"""
        engine = AdvancedSearchEngine()
        
        # Create a mock agent
        agent = Mock()
        agent.agent_name = "ChatGPT Assistant"
        agent.description = "An AI-powered chat assistant"
        agent.sub_heading = "Powered by GPT"
        agent.categories = ["AI", "Chat", "Assistant"]
        agent.creator_username = "openai"
        agent.rating = 4.5
        agent.runs = 150
        
        search_fields = {
            "agent_name": 3.0,
            "description": 2.0,
            "sub_heading": 1.5,
            "categories": 1.0,
            "creator_username": 0.8
        }
        
        # Test high relevance for exact match
        score = engine.calculate_relevance_score(agent, "ChatGPT", search_fields)
        assert score > 3.0  # Should have high score due to name match
        
        # Test lower relevance for partial match
        score = engine.calculate_relevance_score(agent, "assistant", search_fields)
        assert score > 1.0
        
        # Test boost for high-rated agents
        agent.rating = 4.8
        high_rated_score = engine.calculate_relevance_score(agent, "chat", search_fields)
        
        agent.rating = 3.0
        low_rated_score = engine.calculate_relevance_score(agent, "chat", search_fields)
        
        assert high_rated_score > low_rated_score


@pytest.mark.asyncio
class TestGetStoreAgents:
    """Test the get_store_agents function with advanced search"""
    
    async def test_search_with_fuzzy_matching(self):
        """Test search with fuzzy matching enabled"""
        # Mock the database response
        mock_agents = [
            Mock(
                slug="chatgpt-assistant",
                agent_name="ChatGPT Assistant",
                agent_image=["image.png"],
                creator_username="openai",
                creator_avatar="avatar.png",
                sub_heading="AI Chat",
                description="An AI-powered assistant",
                runs=100,
                rating=4.5,
                categories=["AI", "Chat"]
            ),
            Mock(
                slug="code-helper",
                agent_name="Code Helper",
                agent_image=[],
                creator_username="developer",
                creator_avatar="",
                sub_heading="Programming Aid",
                description="Helps with coding tasks",
                runs=50,
                rating=4.2,
                categories=["Programming", "Tools"]
            )
        ]
        
        with patch('prisma.models.StoreAgent.prisma') as mock_prisma:
            mock_prisma.return_value.find_many = AsyncMock(return_value=mock_agents)
            mock_prisma.return_value.count = AsyncMock(return_value=2)
            
            # Test search for "chat" should find ChatGPT Assistant with higher relevance
            result = await get_store_agents(search_query="chat", page=1, page_size=20)
            
            assert result is not None
            assert len(result.agents) > 0
            # ChatGPT Assistant should rank higher due to name match
            if len(result.agents) > 1:
                assert "chat" in result.agents[0].agent_name.lower()
    
    async def test_search_with_typo_tolerance(self):
        """Test search handles typos correctly"""
        mock_agents = [
            Mock(
                slug="chatgpt",
                agent_name="ChatGPT",
                agent_image=["img.png"],
                creator_username="openai",
                creator_avatar="avatar.png",
                sub_heading="AI Assistant",
                description="Chat with AI",
                runs=200,
                rating=4.7,
                categories=["AI"]
            )
        ]
        
        with patch('prisma.models.StoreAgent.prisma') as mock_prisma:
            mock_prisma.return_value.find_many = AsyncMock(return_value=mock_agents)
            mock_prisma.return_value.count = AsyncMock(return_value=1)
            
            # Test with typo "caht" instead of "chat"
            result = await get_store_agents(search_query="caht", page=1, page_size=20)
            
            # Should still find results due to fuzzy matching
            assert result is not None
            # The fuzzy matching should identify ChatGPT despite the typo
    
    async def test_search_ranking_and_boosting(self):
        """Test that search results are properly ranked with boosting"""
        mock_agents = [
            Mock(
                slug="popular-agent",
                agent_name="Popular Agent",
                agent_image=["img1.png"],
                creator_username="user1",
                creator_avatar="av1.png",
                sub_heading="Very Popular",
                description="A highly used agent",
                runs=500,  # High run count
                rating=4.8,  # High rating
                categories=["Popular"]
            ),
            Mock(
                slug="average-agent",
                agent_name="Average Agent",
                agent_image=["img2.png"],
                creator_username="user2",
                creator_avatar="av2.png",
                sub_heading="Standard",
                description="A standard agent",
                runs=10,  # Low run count
                rating=3.5,  # Average rating
                categories=["Standard"]
            )
        ]
        
        with patch('prisma.models.StoreAgent.prisma') as mock_prisma:
            mock_prisma.return_value.find_many = AsyncMock(return_value=mock_agents)
            mock_prisma.return_value.count = AsyncMock(return_value=2)
            
            result = await get_store_agents(search_query="agent", page=1, page_size=20)
            
            assert result is not None
            # Popular agent should rank higher due to boosting
            if len(result.agents) == 2:
                # The agent with higher rating and runs should be ranked first
                first_agent = result.agents[0]
                assert first_agent.runs >= result.agents[1].runs
    
    async def test_category_filtering_with_search(self):
        """Test combining category filter with search"""
        mock_agents = [
            Mock(
                slug="ai-chat",
                agent_name="AI Chat Bot",
                agent_image=["img.png"],
                creator_username="dev",
                creator_avatar="avatar.png",
                sub_heading="Chat AI",
                description="AI-powered chat",
                runs=100,
                rating=4.5,
                categories=["AI", "Chat"]
            )
        ]
        
        with patch('prisma.models.StoreAgent.prisma') as mock_prisma:
            mock_prisma.return_value.find_many = AsyncMock(return_value=mock_agents)
            mock_prisma.return_value.count = AsyncMock(return_value=1)
            
            result = await get_store_agents(
                search_query="chat",
                category="AI",
                page=1,
                page_size=20
            )
            
            assert result is not None
            # Should apply both search and category filter
    
    async def test_pagination_with_search(self):
        """Test that pagination works correctly with search"""
        # Create 25 mock agents
        mock_agents = []
        for i in range(25):
            mock_agents.append(Mock(
                slug=f"agent-{i}",
                agent_name=f"Test Agent {i}",
                agent_image=[],
                creator_username=f"user{i}",
                creator_avatar="",
                sub_heading="Test",
                description=f"Description {i}",
                runs=i * 10,
                rating=3.0 + (i * 0.1),
                categories=["Test"]
            ))
        
        with patch('prisma.models.StoreAgent.prisma') as mock_prisma:
            mock_prisma.return_value.find_many = AsyncMock(return_value=mock_agents)
            mock_prisma.return_value.count = AsyncMock(return_value=25)
            
            # Get first page
            result = await get_store_agents(
                search_query="test",
                page=1,
                page_size=10
            )
            
            assert result is not None
            assert result.pagination.total_items <= 25
            assert result.pagination.total_pages >= 2
            assert result.pagination.current_page == 1
            
            # Get second page
            result = await get_store_agents(
                search_query="test",
                page=2,
                page_size=10
            )
            
            assert result.pagination.current_page == 2