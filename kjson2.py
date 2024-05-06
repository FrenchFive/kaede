import json

# Function to get a message by Message ID from the JSON file
def get_message_by_message_id(file_path, message_id):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for message in data:
                if message.get("Message ID") == message_id:
                    return message
    except FileNotFoundError:
        return None


file_path = "G:/Chan/Documents/zKaede/KAEDE-V2/data.json"  # Replace with the path to your JSON file
message_id_to_find = 1165385830928502914

message = get_message_by_message_id(file_path, message_id_to_find)

if message:
    print("Message found:")
    print(f"Author: {message['Author']}")
    print(f"Message: {message['Message']}")
    print(f"Seed: {message['Seed']}")
    print(f"Message ID: {message['Message ID']}")
else:
    print(f"Message with Message ID {message_id_to_find} not found.")
