import flwr as fl
print('flower import check passed')

fl.server.start_server(config={"num_rounds": 3})

"""
    Train the model, federated!¶
    With both client and server ready, we can now run everything and see federated learning in action. 
    FL systems usually have a server and multiple clients. We therefore have to start the server first.

    1. Run the server
    2. Run the client
    3. Open another termial and run the second client
"""