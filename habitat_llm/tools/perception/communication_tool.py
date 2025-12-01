#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""Simple communication tool that lets agents exchange short text messages."""

import re
from typing import List, Tuple

from habitat_llm.tools import PerceptionTool
from habitat_llm.utils.grammar import FREE_TEXT


class CommunicationTool(PerceptionTool):
    """
    Allows an agent to send messages to, or read messages from, its teammates.
    The actual data transport is handled by the EnvironmentInterface, which
    stores per-agent queues that persist across planner steps.
    """

    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.skill_config = skill_config
        self._read_synonyms = {"", "read", "listen", "check", "receive"}

    def set_environment(self, env_interface):
        self.env_interface = env_interface

    @property
    def description(self) -> str:
        return self.skill_config.description

    def process_high_level_action(
        self, input_query: str, observations: dict
    ) -> Tuple[None, str]:
        super().process_high_level_action(input_query, observations)
        if not self.env_interface:
            raise ValueError("Environment Interface not set for CommunicationTool")

        normalized = (input_query or "").strip()
        lowered = normalized.lower()

        if lowered in self._read_synonyms:
            pending = self.env_interface.consume_agent_messages(self.agent_uid)
            if len(pending) == 0:
                return (
                    None,
                    "No new teammate messages. Updates now flow into your context automatically whenever your partner speaks.",
                )
            formatted = "\n".join(
                [f"Agent_{msg['from']} said: {msg['message']}" for msg in pending]
            )
            return (
                None,
                "Messages are already appended to your context automatically, but here is the latest queue:\n"
                + formatted,
            )

        if normalized == "":
            return None, "Provide a short message to broadcast."

        # Time game: forbid leaking numeric codes over comms; enforce use of write/read tools.
        try:
            game_conf = getattr(getattr(self.env_interface, "conf", None), "game", None)
            game_type = getattr(game_conf, "type", None)
        except Exception:
            game_type = None
        if game_type == "time_game" and re.search(r"\d", normalized):
            return (
                None,
                "Numeric codes must not be sent via CommunicationTool in the time game. Use write_secret_code/read_secret_code instead.",
            )

        self.env_interface.post_agent_message(self.agent_uid, normalized)
        return (
            None,
            f'Message delivered. Your teammate will see "Agent_{self.agent_uid} said: {normalized}" in their context automatically.',
        )

    @property
    def argument_types(self) -> List[str]:
        return [FREE_TEXT]
