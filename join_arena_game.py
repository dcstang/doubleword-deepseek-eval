#!/usr/bin/env python3
"""
Join the NetMind Agent Arena "London Live · Art Showdown" game.

Usage:
    Set ARENA_API_KEY environment variable, then run:
    python join_arena_game.py
"""

import os
import json
import requests
from datetime import datetime

GAME_ID = "b09ed5fd-72ba-457a-965c-56dd16abeddd"
API_BASE = "https://api.arena42.ai/api/v1"

def get_api_key():
    """Get API key from environment."""
    key = os.environ.get("ARENA_API_KEY")
    if not key:
        raise ValueError("ARENA_API_KEY environment variable not set")
    return key

def register_agent(api_key: str) -> dict:
    """Register an agent in the system."""
    print("Registering agent...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "name": f"Claude-Agent-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "description": "DeepSeek evaluation agent"
    }

    response = requests.post(
        f"{API_BASE}/agents",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    agent_data = response.json()
    print(f"✓ Agent registered: {agent_data}")
    return agent_data

def join_game(api_key: str, agent_id: str) -> dict:
    """Join the game as a participant."""
    print(f"Joining game {GAME_ID}...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{API_BASE}/games/{GAME_ID}/participants",
        headers=headers,
        json={"agent_id": agent_id}
    )
    response.raise_for_status()
    participant_data = response.json()
    print(f"✓ Joined game: {participant_data}")
    return participant_data

def submit_entry(api_key: str, agent_id: str, image_url: str) -> dict:
    """Submit an image entry to the game."""
    print(f"Submitting entry with image: {image_url}...")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "agent_id": agent_id,
        "image_url": image_url,
        "description": "Agent's highlight moment in Arena"
    }

    response = requests.post(
        f"{API_BASE}/games/{GAME_ID}/submissions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()
    submission_data = response.json()
    print(f"✓ Entry submitted: {submission_data}")
    return submission_data

def main():
    """Main flow: register, join, submit."""
    api_key = get_api_key()

    try:
        # Register agent
        agent = register_agent(api_key)
        agent_id = agent.get("id") or agent.get("agent_id")

        # Join game
        join_game(api_key, agent_id)

        # Submit entry with a placeholder image
        # Replace with your actual highlight image URL
        image_url = "https://via.placeholder.com/1200x800?text=Claude+Agent+Highlight"
        submit_entry(api_key, agent_id, image_url)

        print("\n✓ Successfully joined and submitted to London Live · Art Showdown!")

    except requests.exceptions.RequestException as e:
        print(f"✗ API Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return 1
    except Exception as e:
        print(f"✗ Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
