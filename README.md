# Different federated learning aggregation strategies with a trusted aggregator.
In this project, I evaluated two new aggregation techniques (FedMax aggregation and FedHmean aggregation) and used a trusted third party for secure aggregation without sending the raw gradient to the server.

# Why trusted aggregation
When the client devices send their raw gradient to the server for aggregation, it can easily leak the client's information. Therefore, in this project, without sending the raw gradient to the server, I sent the raw gradient to a trusted third party, and aggregation happened there.
