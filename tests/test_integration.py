import unittest
import asyncio
import struct
import cbor2
import logging
from client import Client
from protocol import (
    ClientMessageChoice,
    ServerMessageTurn,
    ServerMessageGameFinished,
    GameResultFinished,
)
from sternhalma import Player

# Disable logging for tests to keep output clean, or set to DEBUG for debugging
logging.basicConfig(level=logging.CRITICAL)


class TestClientIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.host = "127.0.0.1"
        self.port = 8888
        self.server_msgs = asyncio.Queue()
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        self.server_task = asyncio.create_task(self.server.serve_forever())

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()
        self.server_task.cancel()
        try:
            await self.server_task
        except asyncio.CancelledError:
            pass

    async def handle_client(self, reader, writer):
        try:
            # 1. Expect Hello
            length_bytes = await reader.readexactly(4)
            length = int.from_bytes(length_bytes, "big")
            data = await reader.readexactly(length)
            msg = cbor2.loads(data)
            await self.server_msgs.put(("received", msg))

            if msg.get("type") != "hello":
                return

            # 2. Send Welcome
            welcome = {
                "type": "welcome",
                "session_id": "test-session",
            }
            self._send(writer, welcome)
            await self.server_msgs.put(("sent", welcome))

            # 3. Send Turn
            # Allow some time for client to be ready to receive
            await asyncio.sleep(0.1)
            turn = {"type": "turn", "movements": [[0, 0, 1, 1], [2, 2, 3, 3]]}
            self._send(writer, turn)
            await self.server_msgs.put(("sent", turn))

            # 4. Expect Choice
            length_bytes = await reader.readexactly(4)
            length = int.from_bytes(length_bytes, "big")
            data = await reader.readexactly(length)
            msg = cbor2.loads(data)
            await self.server_msgs.put(("received", msg))

            if msg.get("type") != "choice":
                return

            # 5. Send Game Finished
            finished = {
                "type": "game_finished",
                "result": {
                    "type": "finished",
                    "winner": 1,
                    "total_turns": 5,
                    "scores": [10, 5],
                },
            }
            self._send(writer, finished)
            await self.server_msgs.put(("sent", finished))

        except Exception as e:
            await self.server_msgs.put(("error", str(e)))
        finally:
            writer.close()
            await writer.wait_closed()

    def _send(self, writer, msg):
        data = cbor2.dumps(msg)
        writer.write(struct.pack(">I", len(data)) + data)

    async def test_game_flow(self):
        """Test a full happy-path game flow"""
        async with Client(self.host, self.port) as client:
            # 1. Handshake verification
            # Client connects and sends Hello automatically in __aenter__
            # check server received Hello
            event, msg = await asyncio.wait_for(self.server_msgs.get(), 1.0)
            self.assertEqual(event, "received")
            self.assertEqual(msg["type"], "hello")

            # Client should verify Welcome inside __aenter__ -> send_message -> receive_message loop?
            # No, client sends Hello in __aenter__, but doesn't wait for Welcome there.
            # It waits for Welcome in assign_player().

            # verify server sent welcome
            event, msg = await asyncio.wait_for(self.server_msgs.get(), 1.0)
            self.assertEqual(event, "sent")
            self.assertEqual(msg["type"], "welcome")

            # 2. Assign Player
            await client.handshake()
            self.assertEqual(client.session_id, "test-session")

            # 3. Game Loop (Turn -> Choice)

            # Verify server sent Turn
            event, msg = await asyncio.wait_for(self.server_msgs.get(), 1.0)
            self.assertEqual(event, "sent")
            self.assertEqual(msg["type"], "turn")

            # Client receives Turn
            server_msg = await client.receive_message()
            self.assertIsInstance(server_msg, ServerMessageTurn)

            # Client sends Choice
            # Simulate agent decision
            choice_idx = 1
            await client.send_message(ClientMessageChoice(choice_idx))

            # Verify server received Choice
            event, msg = await asyncio.wait_for(self.server_msgs.get(), 1.0)
            self.assertEqual(event, "received")
            self.assertEqual(msg["type"], "choice")
            self.assertEqual(msg["movement_index"], choice_idx)

            # 4. Game Finished

            # Verify server sent Finished
            event, msg = await asyncio.wait_for(self.server_msgs.get(), 1.0)
            self.assertEqual(event, "sent")
            self.assertEqual(msg["type"], "game_finished")

            # Client receives finished
            server_msg = await client.receive_message()
            match server_msg:
                case ServerMessageGameFinished(GameResultFinished(winner, _, _)):
                    self.assertEqual(winner, Player.Player1)
                case _:
                    self.fail(
                        f"Expected ServerMessageGameFinished with GameResultFinished, got {server_msg}"
                    )


if __name__ == "__main__":
    unittest.main()
