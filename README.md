# FedMax and FedHmean: Federated learning aggregation strategies with a trusted aggregator
In this project, I evaluated two new aggregation techniques (FedMax aggregation and FedHmean aggregation) and used a trusted third party for secure aggregation without sending the raw gradient to the server.

# Why trusted aggregation
When the client devices send their raw gradient to the server for aggregation, it can easily leak the client's information. Therefore, in this project, without sending the raw gradient to the server, I sent the raw gradient to a trusted third party, and aggregation happened there.
```bash
model_worker1.move(worker_trusted)                             ## send worker1 to trusted worker for secure aggregation 
model_worker2.move(worker_trusted)
model_worker3.move(worker_trusted)
model_worker4.move(worker_trusted)
```
# FedMax (Federated max aggregation)
I calculated the maximum gradient coming from different workers after each round and updated the global model with the maximum gradient.
```bash
fc1_weight = (torch.max(model_worker1.fc1.weight, model_worker2.fc1.weight, model_worker3.fc1.weight, model_worker4.fc1.weight)).get()
fc2_weight = (torch.max(model_worker1.fc2.weight.data,model_worker2.fc2.weight.data,model_worker3.fc2.weight.data,model_worker4.fc2.weight.data)).get()
fc1_bias = (torch.max(model_worker1.fc1.bias.data,model_worker2.fc1.bias.data,model_worker3.fc1.bias.data,model_worker4.fc1.bias.data)).get()
fc2_bias = (torch.max(model_worker1.fc2.bias.data,model_worker2.fc2.bias.data,model_worker3.fc2.bias.data,model_worker4.fc2.bias.data)).get()
```
# FedHmean (Federated harmonic mean)
Another aggregation technique is employed in this project: the federated harmonic mean. In this strategy, the reciprocal of each worker's gradient is summed and divided by the number of workers.
```bash
fc1_weight = ((1/(model_worker1.fc1.weight.data)+1/(model_worker2.fc1.weight.data)+1/(model_worker3.fc1.weight.data)+1/(model_worker4.fc1.weight.data))/4).get() 
fc2_weight = ((1/(model_worker1.fc2.weight.data)+1/(model_worker2.fc2.weight.data)+1/(model_worker3.fc2.weight.data)+1/(model_worker4.fc2.weight.data))/4).get()
fc1_bias = ((1/(model_worker1.fc1.bias.data)+1/(model_worker2.fc1.bias.data)+1/(model_worker3.fc1.bias.data)+1/(model_worker4.fc1.bias.data))/4).get()
fc2_bias = ((1/(model_worker1.fc2.bias.data)+1/(model_worker2.fc2.bias.data)+1/(model_worker3.fc2.bias.data)+1/(model_worker4.fc2.bias.data))/4).get()
```
