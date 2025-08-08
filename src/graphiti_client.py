import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import config


class GraphitiClient:
    """Minimal Graphiti sidecar client.

    - If GRAPHITI_ENABLED and GRAPHITI_SERVER_URL set: send episodes via HTTP.
    - Otherwise, write JSONL episodes to GRAPHITI_EPISODE_DIR for later import.
    """

    def __init__(self) -> None:
        self.enabled = config.GRAPHITI_ENABLED
        self.server_url = config.GRAPHITI_SERVER_URL.rstrip("/") if config.GRAPHITI_SERVER_URL else ""
        self.api_key = config.GRAPHITI_API_KEY
        self.workspace = config.GRAPHITI_WORKSPACE
        self.episode_dir = Path(config.GRAPHITI_EPISODE_DIR)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

    def _episode_payload(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return {
            "workspace": self.workspace,
            "title": title,
            "timestamp": int(time.time()),
            "content": content,
            "metadata": metadata or {},
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
    async def send_episode(self, title: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        payload = self._episode_payload(title, content, metadata)

        if not self.enabled or not self.server_url:
            # Write locally for offline mode
            out_path = self.episode_dir / f"{int(time.time() * 1000)}.jsonl"
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=10) as client:
            url = f"{self.server_url}/api/v1/episodes"
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()


