from dotenv import load_dotenv
import logging 
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext
from livekit.plugins import (
    noise_cancellation,
    silero,
    sarvam,
    google,
    openai,
    )
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from dataclasses import dataclass, field
from typing import Optional
import yaml

logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class UserData:
    object_to_find: str
    user_location: Optional[str] = None
    object_found: bool = False
    object_location: Optional[str] = None
    object_image: Optional[str] = None
    prev_agent: Optional[Agent] = None
    agents: dict[str, Agent] = field(default_factory=dict)

    def summarize(self) -> str:
        data = {
            "object_to_find": self.object_to_find or "nothing",
            "user_location": self.user_location or "unknown",
            "object_found": self.object_found,
            "object_location": self.object_location or "unknown",
            "object_image": self.object_image or "no image",
            "prev_agent": self.prev_agent or "no previous agent"
        }
        return yaml.dump(data)

RunContext_T = RunContext[UserData]


class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # add the previous agent's chat history to the current agent
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True, exclude_function_call=False
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # add an instructions including the user data as assistant message
        chat_ctx.add_message(
            role="system",  # role=system works for OpenAI's LLM and Realtime API
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."

class Greeting(BaseAgent):
    pass


async def entrypoint(ctx: agents.JobContext):

    greeting = Greeting()

    session = AgentSession(
          stt = openai.STT(
        model="gpt-4o-transcribe",
        language="en",
        ),  
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=sarvam.TTS(
            target_language_code="en-IN",
            speaker="hitesh"
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=greeting,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))