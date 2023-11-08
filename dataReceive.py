from replit import db
import json


class DataReceiver:

  def __init__(self):
    self.messages = []

  def receiveData(self, data_string):
    """
        Receive and process a string of data.

        Parameters:
        data_string (str): A string containing the data to be processed.

        The method appends the received data to the 'messages' list and prints the list.
        """
    # Ensure that the data received is a string
    if not isinstance(data_string, str):
      raise ValueError("The data provided is not a string.")

    # Append the received data to the messages list
    self.messages.append(data_string)
    # create a json object, "text" is data_string, "date" is 2023-10-01, "time" is 10:00
    newJson = {"text": data_string, "date": "2023-11-08", "time": "10:00"}

    # insert message to database, the key is equal to the number of keys in database
    db[str(len(db.keys()))] = json.dumps(newJson)

    # Print the current list of messages
    for key in db.keys():
      data = json.loads(db[key])
      print(key + ": " + data["text"])
