import pytest
import tkinter as tk
from unittest.mock import patch, MagicMock
from chatbot import chatbot

@pytest.fixture(scope="class")
def chatbot_app(request):
    root = tk.Tk()
    request.cls.chatbot_app = chatbot.ChatbotApplication(root)

    yield
    request.cls.chatbot_app.close()
    root.destroy()


    def test_chatbot_quit(self):
        self.chatbot_app.open_chatbot()
        self.chatbot_app.user_input.set("quit")
        with patch.object(tk.Entry, "config") as mock_config:
            self.chatbot_app.send_message()
            mock_config.assert_called_once_with(state=tk.DISABLED)
