"""Tests for Flask API"""
import pytest
import json
from unittest.mock import Mock, patch
from src.app import app


@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    """Test home endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'name' in data
    assert 'endpoints' in data


def test_health(client):
    """Test health check"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'


def test_analyze_missing_subject(client):
    """Test analyze endpoint with missing subject"""
    response = client.post('/api/analyze',
                           data=json.dumps({}),
                           content_type='application/json')
    assert response.status_code == 400


@patch('src.app.TwoAgentSystem')
def test_analyze_sync(mock_system, client):
    """Test synchronous analyze endpoint"""
    mock_instance = Mock()
    mock_instance.run.return_value = {
        'subject': 'test',
        'report': 'test report',
        'research_findings': 'test findings'
    }
    mock_system.return_value = mock_instance

    response = client.post('/api/analyze/sync',
                           data=json.dumps({'subject': 'test subject'}),
                           content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'report' in data


def test_list_jobs(client):
    """Test list jobs endpoint"""
    response = client.get('/api/jobs')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'jobs' in data
    assert 'total' in data