import slack_sdk
from slack_sdk.errors import SlackApiError


class SlackMessenger:
    def __init__(self, token, channel_id):
        # Initialize the Slack client with the provided token
        self.client = slack_sdk.WebClient(token=token)
        self.channel_id = channel_id

    def send_message(self, message):
        try:
            # Send a message to the Slack channel
            response = self.client.chat_postMessage(
                channel=self.channel_id,
                text=message
            )
            print(f"Message sent: {response['message']['text']}")
        except SlackApiError as e:
            print(f"Error sending message: {e.response['error']}")

    def send_bulk_messages(self, messages):
        for message in messages:
            self.send_message(message)
